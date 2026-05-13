"""
Export select_element data for training/evaluation.

The export contains:
- all approved favorites
- the most recent N pending_approval favorites

It writes train/eval/full JSON files plus metadata, without overwriting data/.
"""

import argparse
import json
import random
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

from prepare_data import DB_CONFIG, SYSTEM_PROMPT, build_expected_output, extract_user_text_and_image

PROJECT_DIR = Path(__file__).parent.parent


def build_training_sample(row: dict) -> dict | None:
    input_data = row["input_data"]
    if not input_data:
        return None

    expected_obj = build_expected_output(row)
    if expected_obj is None:
        return None
    element_index = expected_obj["element_index"]

    text_content, image_data = extract_user_text_and_image(input_data)
    if not text_content:
        return None

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    user_content = []
    if image_data:
        user_content.append({"type": "image", "image": f"data:image/webp;base64,{image_data[:100]}..."})
    user_content.append({"type": "text", "text": text_content})

    expected_output = json.dumps(expected_obj, ensure_ascii=False)
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": expected_output})

    sample = {
        "source": row["source"],
        "call_id": row["call_id"],
        "llm_call_id": row["id"],
        "approval_status": row["approval_status"],
        "call_created_at": row["call_created_at"].isoformat() if row.get("call_created_at") else None,
        "favorite_created_at": row["favorite_created_at"].isoformat() if row.get("favorite_created_at") else None,
        "favorite_updated_at": row["favorite_updated_at"].isoformat() if row.get("favorite_updated_at") else None,
        "model_name": row.get("model_name"),
        "messages": messages,
        "element_index": element_index,
    }
    if image_data:
        sample["image_base64"] = image_data

    return sample


def fetch_rows(pending_limit: int) -> list[dict]:
    query = """
    WITH approved AS (
        SELECT
            c.id,
            c.call_id,
            c.input_data,
            c.output_data,
            c.model_name,
            c.created_at AS call_created_at,
            f.created_at AS favorite_created_at,
            f.updated_at AS favorite_updated_at,
            f.approval_status,
            f.outputs AS favorite_outputs,
            'approved' AS source,
            coalesce(f.updated_at, f.created_at, c.created_at) AS sort_time
        FROM llm_calls c
        JOIN llm_call_favorites f ON c.call_id = f.call_id
        WHERE c.type = 'select_element'
          AND f.approval_status = 'approved'
    ),
    pending AS (
        SELECT
            c.id,
            c.call_id,
            c.input_data,
            c.output_data,
            c.model_name,
            c.created_at AS call_created_at,
            f.created_at AS favorite_created_at,
            f.updated_at AS favorite_updated_at,
            f.approval_status,
            f.outputs AS favorite_outputs,
            'pending_recent' AS source,
            coalesce(f.updated_at, f.created_at, c.created_at) AS sort_time
        FROM llm_calls c
        JOIN llm_call_favorites f ON c.call_id = f.call_id
        WHERE c.type = 'select_element'
          AND f.approval_status = 'pending_approval'
        ORDER BY coalesce(f.updated_at, f.created_at, c.created_at) DESC
        LIMIT %(pending_limit)s
    )
    SELECT * FROM approved
    UNION ALL
    SELECT * FROM pending
    ORDER BY source, sort_time DESC
    """

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(query, {"pending_limit": pending_limit})
    rows = [dict(row) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_archive(export_dir: Path) -> Path:
    archive_path = export_dir.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(export_dir, arcname=export_dir.name)
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending_limit", type=int, default=1000)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path(args.output_dir) if args.output_dir else PROJECT_DIR / "exports" / f"select_element_{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=True)

    rows = fetch_rows(args.pending_limit)
    samples = []
    skipped = 0
    for row in rows:
        sample = build_training_sample(row)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    random.shuffle(samples)
    eval_size = max(1, int(len(samples) * args.eval_ratio)) if samples else 0
    eval_samples = samples[:eval_size]
    train_samples = samples[eval_size:]

    source_counts = Counter(sample["source"] for sample in samples)
    status_counts = Counter(sample["approval_status"] for sample in samples)
    metadata = {
        "created_at": datetime.now().isoformat(),
        "pending_limit": args.pending_limit,
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "raw_rows": len(rows),
        "parsed_samples": len(samples),
        "skipped_rows": skipped,
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "source_counts": dict(source_counts),
        "approval_status_counts": dict(status_counts),
    }

    write_json(export_dir / "train.json", train_samples)
    write_json(export_dir / "eval.json", eval_samples)
    write_json(export_dir / "full.json", samples)
    write_json(export_dir / "metadata.json", metadata)

    archive_path = create_archive(export_dir)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Export dir: {export_dir}")
    print(f"Archive: {archive_path}")


if __name__ == "__main__":
    main()
