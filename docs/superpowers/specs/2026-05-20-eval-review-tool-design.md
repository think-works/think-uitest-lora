# Eval Review Tool Design

## Problem

ui_single_step eval tasks can have multiple valid answers (e.g., clicking different elements may both achieve the goal). Current eval data has only one ground truth per sample, causing false negatives when the model produces a valid but non-matching answer. Need a tool to review "incorrect" predictions, accept valid alternatives, and update eval data for future evaluation runs.

## Data Format Change

### eval.json

Add optional `accepted_actions` field to each sample:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": [{"type": "image", "image": "..."}, {"type": "text", "text": "..."}]},
    {"role": "assistant", "content": "{\"thinking\": \"...\", \"actions\": [{\"action\": \"click\", \"element_index\": 5}]}"}
  ],
  "image_base64": "...",
  "device_type": "android",
  "accepted_actions": [
    [{"action": "click", "element_index": 5}],
    [{"action": "click", "element_index": 8}]
  ]
}
```

- The original GT actions from `messages[assistant]` are always the first entry in `accepted_actions`
- When a reviewer accepts a prediction, it is appended to `accepted_actions`
- `accepted_actions` is only written when there is at least one alternative (i.e., length >= 2)

## Components

### 1. scripts/review_eval.py

Single-file Python script using `http.server` with embedded HTML/JS. Zero external dependencies.

**CLI interface:**
```bash
python scripts/review_eval.py \
  --eval_results data/ui_single_step/eval_results_20260520_123456.json \
  --eval_data data/ui_single_step/eval.json
```

**HTTP server (port 8765):**

- `GET /` — SPA page for reviewing
- `GET /api/errors` — returns all `correct: false` samples, joined with eval data (including image_base64)
- `POST /api/accept` — body: `{"sample_id": 5, "pred_actions": [...]}`, adds pred to accepted_actions and saves eval.json
- `GET /api/stats` — returns progress (reviewed / total errors)

**Frontend (embedded HTML):**

- Filter: only show `correct: false` samples
- Per-sample card layout:
  - Left: screenshot (base64 rendered as `<img>`)
  - Center: GT actions (formatted JSON, highlighted fields)
  - Right: Pred actions (formatted JSON, diff-highlighted against GT)
  - Bottom: three buttons — Accept (Y), Reject (N), Skip (S)
- Keyboard shortcuts: Y / N / S
- Top bar: progress indicator (X / Y reviewed)
- Auto-advance to next error sample after action
- Saves eval.json after each accept action

### 2. eval_ui_single_step.py changes

Modify `compare_actions` to check against all accepted answers:

```python
def compare_actions(pred, gt, accepted_actions=None):
    if accepted_actions:
        for accepted in accepted_actions:
            if _actions_match(pred, accepted):
                return True
    return _actions_match(pred, gt)

def _actions_match(pred, gt):
    if len(pred) != len(gt):
        return False
    for p, g in zip(pred, gt):
        if {k: v for k, v in p.items() if k != "desc"} != {k: v for k, v in g.items() if k != "desc"}:
            return False
    return True
```

In the eval loop, read `accepted_actions` from eval data and pass it to `compare_actions`.

## File Changes Summary

| File | Action |
|------|--------|
| `scripts/review_eval.py` | **New** — review tool |
| `scripts/eval_ui_single_step.py` | **Modify** — `compare_actions` accepts multiple GT answers |

## Out of Scope

- Bulk import of alternative answers from external sources
- Review of parse-failed samples (no pred to compare)
- Authentication or multi-user support
