"""
从 raw_data.json + system_prompts.json 构建 ui_single_step 训练/评估数据。
输出格式与 select_element 训练数据保持一致。
"""

import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ui_single_step"


def parse_output_content(content: str) -> dict | None:
    """解析 output_data.content，返回 action JSON。"""
    if not content:
        return None
    text = re.sub(r"^\s*```json\s*", "", content)
    text = re.sub(r"\s*```\s*$", "", text)
    text = re.sub(r"<think.*?>.*?</think.*?>", "", text, flags=re.DOTALL)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def extract_element_index(obj: dict | None) -> int | None:
    """从解析后的输出中提取 element_index。"""
    if not obj:
        return None
    if "element_index" in obj:
        return obj["element_index"]
    if "actions" in obj and obj["actions"]:
        return obj["actions"][0].get("element_index")
    return None


def normalize_output_to_actions(obj: dict) -> dict:
    """将单 action 格式统一为 actions 列表格式。

    输入:  {"thinking": "...", "action": "click", "element_index": 5, "desc": "..."}
    输出:  {"thinking": "...", "actions": [{"action": "click", "element_index": 5, "desc": "..."}]}
    """
    if "action" in obj and "actions" not in obj:
        action_fields = {}
        for key in ("action", "element_index", "desc", "text", "direction",
                     "seconds", "app", "url", "reason", "tab_index", "command", "value"):
            if key in obj:
                action_fields[key] = obj[key]
        obj = {"thinking": obj.get("thinking", ""), "actions": [action_fields]}
    return obj


def build_training_sample(record: dict, system_prompts: dict[str, str]) -> dict | None:
    input_data = record.get("input_data")
    output_data = record.get("output_data")
    if not input_data or not output_data:
        return None

    device_type = record.get("device_type", "unknown")
    system_prompt = system_prompts.get(device_type)
    if not system_prompt:
        return None

    # 提取用户内容：操作历史（如有）+ 图片 + 文本
    user_content = []
    image_data = ""

    for msg in input_data:
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            user_content.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    user_content.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:image/"):
                        image_data = url.split(",", 1)[1] if "," in url else ""
                        user_content.append({"type": "image", "image": url})

    if not user_content:
        return None

    # 提取 assistant 输出，去掉代码围栏，统一为 actions 格式
    output_content = output_data.get("content", "")
    if not output_content:
        return None

    parsed = parse_output_content(output_content)
    if parsed is None:
        return None

    parsed = normalize_output_to_actions(parsed)
    output_content = json.dumps(parsed, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_content},
    ]

    element_index = extract_element_index(parsed)

    sample = {
        "call_id": record["call_id"],
        "llm_call_id": record["id"],
        "device_type": device_type,
        "model_name": record.get("model_name"),
        "ref_id": record["ref_id"],
        "created_at": record.get("created_at"),
        "messages": messages,
    }
    if element_index is not None:
        sample["element_index"] = element_index
    if image_data:
        sample["image_base64"] = image_data

    return sample


def main():
    random.seed(42)

    # 加载数据
    with open(DATA_DIR / "raw_data.json", encoding="utf-8") as f:
        raw = json.load(f)
    with open(DATA_DIR / "system_prompts.json", encoding="utf-8") as f:
        system_prompts = json.load(f)

    records = raw["records"]
    print(f"Loaded {len(records)} records from raw_data.json")

    # 构建训练样本
    samples = []
    skipped = 0
    for record in records:
        sample = build_training_sample(record, system_prompts)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    print(f"Parsed {len(samples)} samples, skipped {skipped}")

    # 统计
    device_counts = Counter(s["device_type"] for s in samples)
    has_image = sum(1 for s in samples if "image_base64" in s)
    has_element_index = sum(1 for s in samples if "element_index" in s)
    print(f"Device distribution: {dict(device_counts)}")
    print(f"With image: {has_image}, with element_index: {has_element_index}")

    # 划分训练集和验证集 (90/10)
    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train_samples = samples[:split]
    eval_samples = samples[split:]

    # 保存
    def write_json(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    write_json(DATA_DIR / "train.json", train_samples)
    write_json(DATA_DIR / "eval.json", eval_samples)
    write_json(DATA_DIR / "full.json", samples)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "total_records": len(records),
        "parsed_samples": len(samples),
        "skipped": skipped,
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "device_counts": dict(device_counts),
    }
    write_json(DATA_DIR / "metadata.json", metadata)

    print(f"\nSaved {len(train_samples)} train, {len(eval_samples)} eval samples")
    print(f"Output: {DATA_DIR}")


if __name__ == "__main__":
    main()
