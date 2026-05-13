"""
Export ui_single_step training/evaluation data.

Dedup logic:
1. Query all ui_tasks for May 12-13 (no status filter)
2. Normalize original_task: remove digits, Chinese/English punctuation, newlines, spaces
3. Group by normalized content (same content = same task template)
4. For each group, pick the first task_id that has ui_single_step llm_calls
5. Filter out groups with no ui_single_step records
"""

import json
import re
from collections import defaultdict
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


def normalize_task(original_task: str) -> str:
    """标准化：去掉数字、中英文标点、换行、空格，只保留中文和英文字母。"""
    if not original_task:
        return ""
    return re.sub(r"[^一-鿿a-zA-Z]", "", original_task)


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Step 1: Query all ui_tasks for May 12-13
    cur.execute("""
        SELECT task_id, original_task, status, device_type, created_at
        FROM ui_tasks
        WHERE created_at >= '2026-05-12'
          AND created_at < '2026-05-14'
        ORDER BY created_at
    """)
    tasks = [dict(r) for r in cur.fetchall()]
    print(f"[Step 1] Total ui_tasks (May 12-13): {len(tasks)}")

    # Step 2-3: Normalize and group
    groups = defaultdict(list)
    for task in tasks:
        key = normalize_task(task["original_task"])
        groups[key].append(task)
    print(f"[Step 2-3] Unique normalized groups: {len(groups)}")

    # Step 4: Batch query ui_single_step counts per ref_id
    all_task_ids = [t["task_id"] for t in tasks]
    cur.execute("""
        SELECT ref_id, COUNT(*) AS cnt
        FROM llm_calls
        WHERE type = 'ui_single_step'
          AND ref_id IN %s
        GROUP BY ref_id
    """, (tuple(all_task_ids),))
    ui_step_counts = {r["ref_id"]: r["cnt"] for r in cur.fetchall()}

    # For each group, pick the first task_id that has ui_single_step calls
    representative_task_ids = []
    group_info = []
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
                group_info.append({
                    "normalized_task": norm_key,
                    "representative_task_id": tid,
                    "device_type": task["device_type"],
                    "ui_single_step_count": ui_step_counts[tid],
                    "group_size": len(group_tasks),
                    "original_task": task["original_task"],
                })
                found = True
                break

        if not found:
            no_records_count += 1

    print(f"[Step 4] Representative tasks with ui_single_step: {len(representative_task_ids)}")
    print(f"[Step 5] Filtered out (no ui_single_step): {no_records_count}")

    # Step 5: Batch get all ui_single_step llm_calls for representative tasks
    # Build task_id -> device_type mapping
    task_device_map = {tid: gi["device_type"] for gi in group_info for tid in [gi["representative_task_id"]]}

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

    cur.close()
    conn.close()

    # Export
    export_dir = PROJECT_DIR / "data" / "ui_single_step"
    export_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "exported_at": datetime.now().isoformat(),
        "total_tasks": len(tasks),
        "unique_groups": len(groups),
        "filtered_out": no_records_count,
        "representative_tasks": len(representative_task_ids),
        "total_records": len(records),
        "group_info": group_info,
        "records": records,
    }

    output_path = export_dir / "raw_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nExported {len(records)} ui_single_step records to {output_path}")


if __name__ == "__main__":
    main()
