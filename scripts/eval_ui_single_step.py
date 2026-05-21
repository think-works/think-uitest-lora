"""
评估脚本: 对 ui_single_step eval 数据集推理，比较 actions（忽略 desc 字段）是否一致。

用法:
    python scripts/eval_ui_single_step.py                           # 评估原始模型
    python scripts/eval_ui_single_step.py --model_path output/final # 评估微调后模型
"""

import argparse
import base64
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ui_single_step"

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"


def load_system_prompts():
    with open(DATA_DIR / "system_prompts.json", encoding="utf-8") as f:
        return json.load(f)


def parse_actions(text: str) -> list[dict] | None:
    """从模型输出中解析 actions 列表。"""
    cleaned = re.sub(r"<think.*?>.*?</think.*?>", "", text, flags=re.DOTALL)
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not json_match:
        return None
    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    if "actions" in obj:
        return obj["actions"]
    if "action" in obj:
        # Flat action at top level — collect all fields except meta keys
        meta_keys = {"thinking", "actions"}
        action_fields = {k: v for k, v in obj.items() if k not in meta_keys}
        return [action_fields]
    return None


def compare_actions(pred: list[dict], gt: list[dict], accepted_actions: list | None = None) -> bool:
    """比较 pred 与 gt（及可选的 accepted_actions），忽略 desc 字段。"""
    if accepted_actions:
        for accepted in accepted_actions:
            if _actions_match(pred, accepted):
                return True
    return _actions_match(pred, gt)


def _actions_match(pred: list[dict], gt: list[dict]) -> bool:
    if len(pred) != len(gt):
        return False
    for p, g in zip(pred, gt):
        p_filtered = {k: v for k, v in p.items() if k != "desc"}
        g_filtered = {k: v for k, v in g.items() if k != "desc"}
        if p_filtered != g_filtered:
            return False
    return True


def parse_gt_actions(assistant_content: str) -> list[dict]:
    """从 ground truth assistant content 中解析 actions。"""
    obj = json.loads(assistant_content)
    return obj.get("actions", [])


def load_eval_data():
    with open(DATA_DIR / "eval.json", encoding="utf-8") as f:
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


def prepare_inputs(item, processor, system_prompts, img_dir, device):
    messages_raw = item["messages"]
    device_type = item.get("device_type", "unknown")

    system_prompt = system_prompts.get(device_type, system_prompts.get("android", ""))

    gt_actions = []
    user_text = ""
    for msg in messages_raw:
        if msg["role"] == "assistant":
            gt_actions = parse_gt_actions(msg["content"])
        elif msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        user_text += part["text"]
            else:
                user_text = content

    image_base64 = item.get("image_base64", "")
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

    return inputs, gt_actions


def _log(msg: str):
    import sys
    print(msg, flush=True, file=sys.stderr if sys.stderr.isatty() else sys.stdout)


def run_eval(model_path: str, use_4bit: bool = True, limit: int | None = None,
             max_pixels: int | None = None, merge_lora: bool = False):
    _log(f"[1/7] Loading model from: {model_path}")
    adapter_config = Path(model_path) / "adapter_config.json"
    is_adapter = adapter_config.exists()
    base_model_path = DEFAULT_MODEL
    if is_adapter:
        with open(adapter_config, encoding="utf-8") as f:
            base_model_path = json.load(f).get("base_model_name_or_path") or DEFAULT_MODEL
        _log(f"Detected LoRA adapter. Base model: {base_model_path}")
    load_path = base_model_path if is_adapter else model_path

    _log("[2/7] Loading base model weights ...")
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
    _log("[2/7] Base model loaded.")

    if is_adapter:
        if merge_lora:
            _log("[3/7] Merging LoRA adapter into base model ...")
            from peft import PeftModel as _PeftModel
            model = _PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
            _log("[3/7] LoRA merged and unloaded.")
        else:
            _log("[3/7] Loading LoRA adapter (use --merge_lora if this hangs) ...")
            model = PeftModel.from_pretrained(model, model_path)
            _log("[3/7] LoRA adapter loaded.")
        model.eval()

    _log("[4/7] Loading processor ...")
    processor = AutoProcessor.from_pretrained(
        model_path if not is_adapter else load_path,
        trust_remote_code=True,
    )
    if max_pixels is not None:
        processor.image_processor.size["longest_edge"] = max_pixels
    _log("[4/7] Processor loaded.")

    _log("[5/7] Loading eval data ...")
    system_prompts = load_system_prompts()
    eval_data = load_eval_data()
    if limit is not None and limit > 0:
        eval_data = eval_data[:limit]
    _log(f"[5/7] Loaded {len(eval_data)} eval samples.")
    img_dir = DATA_DIR / "images"
    device = model.device

    results = []
    correct = 0
    total = 0
    parse_fail = 0
    start_time = time.time()
    eval_len = len(eval_data)

    action_type_stats = Counter()
    action_type_correct = Counter()

    print(f"\nEvaluating {eval_len} samples ...")

    for i, item in enumerate(eval_data):
        inputs, gt_actions = prepare_inputs(item, processor, system_prompts, img_dir, device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        actual_input_len = inputs["attention_mask"][0].sum().item()
        new_tokens = output_ids.shape[1] - actual_input_len
        output_text = processor.decode(output_ids[0][actual_input_len:], skip_special_tokens=True)

        pred_actions = parse_actions(output_text)

        total += 1
        accepted_actions = item.get("accepted_actions")

        if pred_actions is None:
            parse_fail += 1
            is_correct = False
        else:
            is_correct = compare_actions(pred_actions, gt_actions, accepted_actions)

        if is_correct:
            correct += 1

        for a in gt_actions:
            action_type = str(a.get("action", "unknown"))
            action_type_stats[action_type] += 1
            if is_correct:
                action_type_correct[action_type] += 1

        if i < 3:
            print(f"  RAW ({new_tokens} tokens): {output_text[:300]!r}")

        results.append({
            "sample_id": i,
            "device_type": item.get("device_type"),
            "gt_actions": gt_actions,
            "pred_actions": pred_actions,
            "new_tokens": new_tokens,
            "raw_output": output_text[:500],
            "correct": is_correct,
        })

        elapsed = time.time() - start_time
        status = "✓" if is_correct else "✗"
        print(f"  [{total}/{eval_len}] {status} GT={json.dumps(gt_actions, ensure_ascii=False)[:80]} "
              f"Pred={json.dumps(pred_actions, ensure_ascii=False)[:80] if pred_actions else 'PARSE_FAIL'} "
              f"({elapsed:.1f}s, avg {elapsed/total:.1f}s/sample)")

    accuracy = correct / total if total > 0 else 0
    parse_rate = (total - parse_fail) / total if total > 0 else 0

    total_time = time.time() - start_time

    # Save results first so the file is written even if printing crashes
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = DATA_DIR / f"eval_results_{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump({
            "model_path": model_path,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "parse_fail": parse_fail,
            "action_type_stats": dict(action_type_stats),
            "action_type_correct": dict(action_type_correct),
            "details": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"Detailed results saved to: {result_path}")

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Total samples:  {total}")
    print(f"  Correct:        {correct}")
    print(f"  Parse failed:   {parse_fail}")
    print(f"  Accuracy:       {accuracy:.2%}")
    print(f"  Parse rate:     {parse_rate:.2%}")
    print(f"  Total time:     {total_time:.1f}s")
    if total > 0:
        print(f"  Avg time:       {total_time/total:.1f}s/sample")
    print(f"\n  Accuracy by action type:")
    for action_type in sorted(action_type_stats.keys(), key=str):
        cnt = action_type_stats[action_type]
        acc = action_type_correct[action_type] / cnt if cnt > 0 else 0
        print(f"    {action_type}: {action_type_correct[action_type]}/{cnt} ({acc:.2%})")
    print(f"{'='*60}")

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
