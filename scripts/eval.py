"""
评估脚本: 加载模型，对 eval 数据集推理，比较 element_index 准确率。

注意: 由于 Qwen VL 模型在 batch>1 时 image token 位置编码存在 bug，
      每条样本独立推理（等效 batch_size=1），确保结果正确。

用法:
    python scripts/eval.py                           # 评估原始模型
    python scripts/eval.py --model_path output/final # 评估微调后模型
"""

import argparse
import base64
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"


def load_system_prompt():
    from prepare_data import SYSTEM_PROMPT
    return SYSTEM_PROMPT


def parse_element_index(text: str) -> int | None:
    text = re.sub(r"<thinkitype>.*?</thinkitype>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinkitype_keyword>.*?</thinkitype_keyword>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think.*?>.*?</think.*?>", "", text, flags=re.DOTALL)

    json_match = re.search(r"\{[^{}]*\}", text)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            idx = obj.get("element_index") or obj.get("index")
            if idx is not None:
                return int(idx)
        except (json.JSONDecodeError, ValueError):
            pass

    m = re.search(r'"element_index"\s*:\s*(-?\d+)', text)
    if m:
        return int(m.group(1))

    m = re.search(r'"index"\s*:\s*(-?\d+)', text)
    if m:
        return int(m.group(1))

    return None


def load_eval_data():
    eval_path = DATA_DIR / "eval.json"
    with open(eval_path) as f:
        return json.load(f)


def prepare_image(image_base64: str, img_dir: Path) -> str | None:
    if not image_base64:
        return None
    img_dir.mkdir(exist_ok=True)
    img_hash = hashlib.md5(image_base64[:1000].encode()).hexdigest()[:12]
    img_path = img_dir / f"{img_hash}.webp"
    if not img_path.exists():
        img_bytes = base64.b64decode(image_base64)
        img_path.write_bytes(img_bytes)
    return str(img_path)


def prepare_inputs(item, processor, system_prompt, img_dir, device):
    messages_raw = item["messages"]
    gt_index = item.get("element_index", None)
    image_base64 = item.get("image_base64", "")

    user_text = ""
    for msg in messages_raw:
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        user_text += part["text"]
            else:
                user_text = content

    pil_image = None
    if image_base64:
        img_path = prepare_image(image_base64, img_dir)
        if img_path:
            try:
                pil_image = Image.open(img_path).convert("RGB")
            except Exception:
                pass

    user_content = []
    if pil_image:
        user_content.append({"type": "image", "image": pil_image})
    user_content.append({"type": "text", "text": user_text})

    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    text_prompt = processor.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    images_arg = [pil_image] if pil_image else None
    inputs = processor(
        text=[text_prompt],
        images=images_arg,
        return_tensors="pt",
        padding=True,
    ).to(device)

    return inputs, gt_index


def run_eval(model_path: str, use_4bit: bool = True, limit: int | None = None,
             max_pixels: int | None = None):
    print(f"Loading model from: {model_path}")
    adapter_config = Path(model_path) / "adapter_config.json"
    is_adapter = adapter_config.exists()
    base_model_path = DEFAULT_MODEL
    if is_adapter:
        with open(adapter_config, encoding="utf-8") as f:
            base_model_path = json.load(f).get("base_model_name_or_path") or DEFAULT_MODEL
        print(f"Detected LoRA adapter. Base model: {base_model_path}")
    load_path = base_model_path if is_adapter else model_path

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            load_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    if is_adapter:
        model = PeftModel.from_pretrained(model, model_path)
        model.eval()

    processor = AutoProcessor.from_pretrained(
        model_path if not is_adapter else load_path,
        trust_remote_code=True,
    )
    if max_pixels is not None:
        processor.image_processor.size["longest_edge"] = max_pixels

    system_prompt = load_system_prompt()
    eval_data = load_eval_data()
    if limit is not None and limit > 0:
        eval_data = eval_data[:limit]
    img_dir = DATA_DIR / "images"
    device = model.device

    results = []
    correct = 0
    total = 0
    parse_fail = 0
    start_time = time.time()
    eval_len = len(eval_data)

    print(f"\nEvaluating {eval_len} samples ...")

    for i, item in enumerate(eval_data):
        inputs, gt_index = prepare_inputs(item, processor, system_prompt, img_dir, device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        actual_input_len = inputs["attention_mask"][0].sum().item()
        new_tokens = output_ids.shape[1] - actual_input_len
        output_text = processor.decode(output_ids[0][actual_input_len:], skip_special_tokens=True)
        if i < 3:
            print(f"  RAW ({new_tokens} tokens): {output_text[:300]!r}")
        pred_index = parse_element_index(output_text)

        is_correct = pred_index == gt_index
        total += 1
        if is_correct:
            correct += 1
        if pred_index is None:
            parse_fail += 1

        results.append({
            "sample_id": i,
            "gt_element_index": gt_index,
            "pred_element_index": pred_index,
            "new_tokens": new_tokens,
            "raw_output": output_text[:500],
            "correct": is_correct,
        })

        elapsed = time.time() - start_time
        status = "✓" if is_correct else "✗"
        print(f"  [{total}/{eval_len}] {status} GT={gt_index} Pred={pred_index} tokens={new_tokens}  ({elapsed:.1f}s elapsed, avg {elapsed/total:.1f}s/sample)")

    accuracy = correct / total if total > 0 else 0
    parse_rate = (total - parse_fail) / total if total > 0 else 0

    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Total samples: {total}")
    print(f"  Correct:       {correct}")
    print(f"  Parse failed:  {parse_fail}")
    print(f"  Accuracy:      {accuracy:.2%}")
    print(f"  Parse rate:    {parse_rate:.2%}")
    print(f"  Total time:    {total_time:.1f}s")
    print(f"  Avg time:      {total_time/total:.1f}s/sample" if total > 0 else "  Avg time:      N/A")
    print(f"{'='*50}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = DATA_DIR / f"eval_results_{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump({
            "model_path": model_path,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "parse_fail": parse_fail,
            "details": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"Detailed results saved to: {result_path}")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL, help="Model path or HuggingFace model ID")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--limit", type=int, default=None, help="Limit eval samples for smoke tests")
    parser.add_argument("--max_pixels", type=int, default=None, help="Optional image max pixels, e.g. 1003520")
    args = parser.parse_args()

    run_eval(args.model_path, use_4bit=not args.no_4bit, limit=args.limit,
             max_pixels=args.max_pixels)


if __name__ == "__main__":
    main()
