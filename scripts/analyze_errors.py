"""
分析 eval 错误样本。

用法:
    python scripts/analyze_errors.py                           # 分析最新的 eval_results_*.json
    python scripts/analyze_errors.py data/eval_results_xxx.json  # 指定文件
"""

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def find_latest_results():
    files = sorted(DATA_DIR.glob("eval_results_*.json"))
    if not files:
        files = sorted(DATA_DIR.glob("eval_results.json"))
    if not files:
        print("No eval results found.")
        sys.exit(1)
    return files[-1]


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = find_latest_results()
    print(f"Analyzing: {path}\n")

    with open(path) as f:
        data = json.load(f)

    details = data["details"]
    total = len(details)
    errors = [d for d in details if not d["correct"]]
    parse_fails = [d for d in errors if d["pred_element_index"] is None]
    wrong_preds = [d for d in errors if d["pred_element_index"] is not None]

    print(f"Total: {total}  |  Errors: {len(errors)}  |  Parse fail: {len(parse_fails)}  |  Wrong prediction: {len(wrong_preds)}")
    print(f"Accuracy: {data['accuracy']:.2%}\n")

    # Token length stats
    if details and "new_tokens" in details[0]:
        token_lens = [d["new_tokens"] for d in details]
        token_lens_correct = [d["new_tokens"] for d in details if d["correct"]]
        token_lens_error = [d["new_tokens"] for d in details if not d["correct"]]
        print(f"Generated tokens (all):      avg={sum(token_lens)/len(token_lens):.1f}  min={min(token_lens)}  max={max(token_lens)}")
        if token_lens_correct:
            print(f"Generated tokens (correct):  avg={sum(token_lens_correct)/len(token_lens_correct):.1f}  min={min(token_lens_correct)}  max={max(token_lens_correct)}")
        if token_lens_error:
            print(f"Generated tokens (error):    avg={sum(token_lens_error)/len(token_lens_error):.1f}  min={min(token_lens_error)}  max={max(token_lens_error)}")
        print()

    # Parse failures
    if parse_fails:
        print(f"=== Parse Failures ({len(parse_fails)}) ===")
        for d in parse_fails[:20]:
            print(f"  sample={d['sample_id']} GT={d['gt_element_index']}  raw={d['raw_output']!r}")
        print()

    # Wrong predictions - show offset distribution
    if wrong_preds:
        print(f"=== Wrong Predictions ({len(wrong_preds)}) ===")
        offsets = {}
        for d in wrong_preds:
            off = d["pred_element_index"] - d["gt_element_index"]
            offsets[off] = offsets.get(off, 0) + 1
        print("  Offset (pred - GT) distribution:")
        for off in sorted(offsets.keys()):
            bar = "█" * offsets[off]
            print(f"    {off:+4d}: {offsets[off]:3d} {bar}")
        print()

        print("  Sample details (first 30):")
        for d in wrong_preds[:30]:
            off = d["pred_element_index"] - d["gt_element_index"]
            print(f"  sample={d['sample_id']:3d} GT={d['gt_element_index']:3d} Pred={d['pred_element_index']:3d} off={off:+4d}  raw={d['raw_output']!r}")


if __name__ == "__main__":
    main()
