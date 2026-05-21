"""
从数据库导出 ui_single_step 数据并直接处理为训练/评估数据。

流程:
1. 查询最近 2 个月的 ui_tasks
2. 按 normalized original_task 去重，每组取一个代表 task
3. 筛选 model_name 以 gemini 开头且同一 task 下模型一致的 llm_calls
4. 解析 input/output 构建训练样本，按 90/10 划分 train/eval

用法:
    python scripts/export_ui_single_step.py               # dry-run，仅打印统计
    python scripts/export_ui_single_step.py --write        # 导出 train.json / eval.json
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    "host": "10.0.3.119",
    "port": 5432,
    "dbname": "think_moss",
    "user": "postgres",
    "password": "Think@123",
}

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ui_single_step"


# ── DB 查询 ──────────────────────────────────────────────────────────────────


def normalize_task(original_task: str) -> str:
    if not original_task:
        return ""
    return re.sub(r"[^一-鿿a-zA-Z]", "", original_task)


def fetch_records(cur) -> tuple[list[dict], dict]:
    """从数据库查询去重后的 llm_calls 记录，返回 (records, stats)。"""
    cur.execute("""
        SELECT task_id, original_task, status, device_type, created_at
        FROM ui_tasks
        WHERE created_at >= CURRENT_DATE - INTERVAL '2 months'
        ORDER BY created_at
    """)
    tasks = [dict(r) for r in cur.fetchall()]

    groups = defaultdict(list)
    for task in tasks:
        groups[normalize_task(task["original_task"])].append(task)

    all_task_ids = [t["task_id"] for t in tasks]
    if not all_task_ids:
        return [], {"total_tasks": 0, "unique_groups": 0, "filtered_out": 0,
                     "representative_tasks": 0, "skipped_mixed": 0}

    cur.execute("""
        SELECT ref_id, COUNT(*) AS cnt, MIN(model_name) AS model_name
        FROM llm_calls
        WHERE type = 'ui_single_step'
          AND ref_id IN %s
          AND model_name LIKE 'gemini%%'
        GROUP BY ref_id
        HAVING COUNT(DISTINCT model_name) = 1
    """, (tuple(all_task_ids),))
    ui_step_counts = {r["ref_id"]: r["cnt"] for r in cur.fetchall()}

    cur.execute("""
        SELECT ref_id
        FROM llm_calls
        WHERE type = 'ui_single_step'
          AND ref_id IN %s
          AND model_name LIKE 'gemini%%'
        GROUP BY ref_id
        HAVING COUNT(DISTINCT model_name) > 1
    """, (tuple(all_task_ids),))
    skipped_mixed = len(cur.fetchall())

    representative_task_ids = []
    task_device_map = {}
    no_records_count = 0

    for norm_key, group_tasks in groups.items():
        if not norm_key:
            no_records_count += 1
            continue
        found = False
        for task in group_tasks:
            tid = task["task_id"]
            if tid in ui_step_counts:
                representative_task_ids.append(tid)
                task_device_map[tid] = task["device_type"]
                found = True
                break
        if not found:
            no_records_count += 1

    records = []
    if representative_task_ids:
        cur.execute("""
            SELECT id, call_id, type, input_data, output_data, model_name,
                   ref_id, prompt_tokens, completion_tokens, total_tokens,
                   duration_ms, created_at
            FROM llm_calls
            WHERE type = 'ui_single_step'
              AND ref_id IN %s
            ORDER BY ref_id, created_at
        """, (tuple(representative_task_ids),))
        for row in cur.fetchall():
            r = dict(row)
            if r.get("created_at"):
                r["created_at"] = r["created_at"].isoformat()
            r["device_type"] = task_device_map.get(r["ref_id"], "unknown")
            records.append(r)

    stats = {
        "total_tasks": len(tasks),
        "unique_groups": len(groups),
        "filtered_out": no_records_count,
        "representative_tasks": len(representative_task_ids),
        "skipped_mixed": skipped_mixed,
    }
    return records, stats


# ── 样本构建 ──────────────────────────────────────────────────────────────────


def parse_output_content(content: str | list) -> dict | None:
    if not content:
        return None
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        content = "\n".join(parts)
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


def normalize_output_to_actions(obj: dict) -> dict:
    if "action" in obj and "actions" not in obj:
        meta_keys = {"thinking", "actions"}
        action_fields = {k: v for k, v in obj.items() if k not in meta_keys}
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

    element_index = None
    if "element_index" in parsed:
        element_index = parsed["element_index"]
    elif "actions" in parsed and parsed["actions"]:
        element_index = parsed["actions"][0].get("element_index")

    sample = {
        "device_type": device_type,
        "messages": messages,
    }
    if element_index is not None:
        sample["element_index"] = element_index
    if image_data:
        sample["image_base64"] = image_data
    return sample


# ── 主流程 ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="导出 train.json / eval.json（默认 dry-run）")
    args = parser.parse_args()

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    records, stats = fetch_records(cur)
    cur.close()
    conn.close()

    print(f"[DB] tasks={stats['total_tasks']} groups={stats['unique_groups']} "
          f"representative={stats['representative_tasks']} "
          f"filtered={stats['filtered_out']} mixed_model={stats['skipped_mixed']}")
    print(f"[DB] {len(records)} llm_calls fetched")

    if not args.write:
        print(f"\n[dry-run] Use --write to export train/eval data.")
        return

    with open(DATA_DIR / "system_prompts.json", encoding="utf-8") as f:
        system_prompts = json.load(f)

    samples = []
    skipped = 0
    for record in records:
        sample = build_training_sample(record, system_prompts)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    device_counts = Counter(s["device_type"] for s in samples)
    has_image = sum(1 for s in samples if "image_base64" in s)
    has_element_index = sum(1 for s in samples if "element_index" in s)
    print(f"[Parse] {len(samples)} samples, {skipped} skipped")
    print(f"[Stats] devices={dict(device_counts)} image={has_image} element_index={has_element_index}")

    random.seed(42)
    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train_samples = samples[:split]
    eval_samples = samples[split:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def write_json(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    write_json(DATA_DIR / "train.json", train_samples)
    write_json(DATA_DIR / "eval.json", eval_samples)

    print(f"\nSaved {len(train_samples)} train, {len(eval_samples)} eval -> {DATA_DIR}")


if __name__ == "__main__":
    main()
