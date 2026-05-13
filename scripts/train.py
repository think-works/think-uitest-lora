"""
LoRA/QLoRA fine-tuning for Qwen/Qwen3.6-35B-A3B on the select_element task.

The dataset is multimodal: each sample contains one screenshot plus the
element list text. This script keeps the image path in the dataset and uses a
custom collator so the processor can build model inputs for both text and image.
"""

import argparse
import base64
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
IGNORE_INDEX = -100


def image_base64_to_path(image_base64: str, base_dir: Path) -> str:
    """Persist a base64 image under base_dir/images and return its path."""
    img_dir = base_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    img_hash = hashlib.md5(image_base64[:1000].encode()).hexdigest()[:12]
    img_path = img_dir / f"{img_hash}.webp"
    if not img_path.exists():
        img_path.write_bytes(base64.b64decode(image_base64))
    return str(img_path)


def load_json_dataset(path: Path, max_chars: int | None = None) -> Dataset:
    """Load prepared JSON samples and materialize image paths for training."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    base_dir = path.parent
    records = []
    for item in raw:
        image_path = ""
        if item.get("image_base64"):
            image_path = image_base64_to_path(item["image_base64"], base_dir)
        messages_json = json.dumps(item["messages"], ensure_ascii=False)
        if max_chars is not None and len(messages_json) > max_chars:
            continue

        records.append(
            {
                "messages": messages_json,
                "image_path": image_path,
                "element_index": item.get("element_index", -1),
                "length": len(messages_json),
            }
        )

    return Dataset.from_list(records)


def split_messages(messages: list[dict]) -> tuple[list[dict], str]:
    """Return prompt messages and the assistant target string."""
    prompt_messages = []
    assistant_text = None
    for message in messages:
        if message["role"] == "assistant":
            assistant_text = message["content"]
            break
        prompt_messages.append(message)

    if assistant_text is None:
        raise ValueError("Training sample has no assistant message")
    return prompt_messages, assistant_text


def attach_image(messages: list[dict], image: Image.Image | None) -> list[dict]:
    """Replace the stored image placeholder with the PIL image expected by Qwen."""
    converted = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") in {"image", "image_url"}:
                    if image is not None:
                        parts.append({"type": "image", "image": image})
                else:
                    parts.append(part)
            converted.append({"role": message["role"], "content": parts})
        else:
            converted.append(message)
    return converted


class QwenSelectElementCollator:
    """Build multimodal inputs and mask prompt tokens from the loss."""

    def __init__(self, processor, max_length: int, logits_to_keep: int):
        self.processor = processor
        self.max_length = max_length
        self.logits_to_keep = logits_to_keep

    @staticmethod
    def _find_last_subsequence(sequence: list[int], pattern: list[int]) -> int | None:
        if not pattern or len(pattern) > len(sequence):
            return None
        for start in range(len(sequence) - len(pattern), -1, -1):
            if sequence[start : start + len(pattern)] == pattern:
                return start
        return None

    def _encode_one(self, sample: dict) -> dict[str, torch.Tensor]:
        messages = sample["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        image = None
        image_path = sample.get("image_path")
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")

        prompt_messages, assistant_text = split_messages(messages)
        prompt_messages = attach_image(prompt_messages, image)
        full_messages = prompt_messages + [{"role": "assistant", "content": assistant_text}]

        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        answer_start_char = full_text.rfind(assistant_text)
        if answer_start_char < 0:
            raise ValueError("Could not locate assistant answer in formatted chat")

        image_arg = [image] if image is not None else None
        full = self.processor(
            text=[full_text],
            images=image_arg,
            return_tensors="pt",
            truncation=False,
        )

        item = {}
        for key, value in full.items():
            if key in {"input_ids", "attention_mask", "mm_token_type_ids"}:
                item[key] = value.squeeze(0)
            else:
                item[key] = value
        if item["input_ids"].shape[0] > self.max_length:
            raise ValueError(
                f"Sample length {item['input_ids'].shape[0]} exceeds max_length={self.max_length}. "
                "Increase --max_length or implement structure-aware truncation that preserves image tokens."
            )

        labels = item["input_ids"].clone()
        answer_tokens = self.processor.tokenizer(
            assistant_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        answer_start = self._find_last_subsequence(item["input_ids"].tolist(), answer_tokens)
        if answer_start is None:
            answer_prefix_text = full_text[:answer_start_char]
            answer_prefix = self.processor(
                text=[answer_prefix_text],
                images=image_arg,
                return_tensors="pt",
                truncation=False,
            )
            answer_start = min(answer_prefix["input_ids"].shape[1], labels.shape[0])

        labels[:answer_start] = IGNORE_INDEX
        labels[item["attention_mask"] == 0] = IGNORE_INDEX
        item["labels"] = labels
        return item

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        encoded = [self._encode_one(sample) for sample in batch]

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id

        max_len = max(item["input_ids"].shape[0] for item in encoded)
        output = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        if any("mm_token_type_ids" in item for item in encoded):
            output["mm_token_type_ids"] = []

        for item in encoded:
            pad_len = max_len - item["input_ids"].shape[0]
            output["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=pad_id))
            output["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0))
            output["labels"].append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=IGNORE_INDEX))
            if "mm_token_type_ids" in output:
                token_types = item.get("mm_token_type_ids")
                if token_types is None:
                    token_types = torch.zeros_like(item["input_ids"])
                output["mm_token_type_ids"].append(torch.nn.functional.pad(token_types, (0, pad_len), value=0))

        batch_out = {key: torch.stack(value) for key, value in output.items()}
        logits_to_keep = min(self.logits_to_keep, batch_out["input_ids"].shape[1])
        batch_out["labels"] = batch_out["labels"][:, -logits_to_keep:]
        batch_out["logits_to_keep"] = logits_to_keep

        for key in ("pixel_values", "image_grid_thw"):
            values = [item[key] for item in encoded if key in item]
            if values:
                batch_out[key] = torch.cat(values, dim=0)

        return batch_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--data_dir", default=str(DATA_DIR), help="Directory containing train.json and eval.json.")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max_length", type=int, default=65536)
    parser.add_argument("--max_chars", type=int, default=None, help="Skip samples whose serialized messages exceed this many chars.")
    parser.add_argument("--logits_to_keep", type=int, default=256, help="Only compute LM logits for the final N tokens to reduce VRAM.")
    parser.add_argument("--min_pixels", type=int, default=3136)
    parser.add_argument("--max_pixels", type=int, default=1003520)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument(
        "--lora_targets",
        choices=["attention", "all"],
        default="all",
        help="attention is faster; all also trains MLP projections and may fit better with more data.",
    )
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--deepspeed", default=None, help="Path to DeepSpeed config. Use with --no_4bit for ZeRO-3 bf16 LoRA.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Set by DeepSpeed launcher.")
    parser.add_argument("--eval_loss", action="store_true", help="Run Trainer eval loss during training. Can OOM on long VL samples.")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Faster but uses more VRAM.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    print(f"Loading model: {args.model_name}")
    print(f"Output dir: {output_dir}")

    using_deepspeed = args.deepspeed is not None
    if using_deepspeed and not args.no_4bit:
        raise ValueError("DeepSpeed ZeRO-3 should be run with --no_4bit for this script.")

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = {
        "quantization_config": quantization_config,
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if not using_deepspeed:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.lora_targets == "all":
        target_modules += ["gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    processor.image_processor.size["shortest_edge"] = args.min_pixels
    processor.image_processor.size["longest_edge"] = args.max_pixels
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    data_dir = Path(args.data_dir)
    train_dataset = load_json_dataset(data_dir / "train.json", max_chars=args.max_chars)
    eval_dataset = load_json_dataset(data_dir / "eval.json", max_chars=args.max_chars if args.eval_loss else None)
    print(f"Train samples: {len(train_dataset)}, eval samples: {len(eval_dataset)}")

    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "tf32": True,
        "logging_steps": 5,
        "eval_strategy": "epoch" if args.eval_loss else "no",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": args.eval_loss,
        "metric_for_best_model": "eval_loss" if args.eval_loss else None,
        "greater_is_better": False if args.eval_loss else None,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
    }
    if args.deepspeed:
        training_kwargs["deepspeed"] = args.deepspeed
    training_arg_names = inspect.signature(TrainingArguments.__init__).parameters
    if "group_by_length" in training_arg_names:
        training_kwargs["group_by_length"] = True
        training_kwargs["length_column_name"] = "length"
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=QwenSelectElementCollator(
            processor,
            max_length=args.max_length,
            logits_to_keep=args.logits_to_keep,
        ),
    )

    print("Starting training...")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"LoRA adapter saved to: {final_dir}")


if __name__ == "__main__":
    main()
