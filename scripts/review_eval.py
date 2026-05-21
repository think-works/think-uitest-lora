"""
Eval Review Tool — web UI for reviewing incorrect eval predictions.

Usage:
    python scripts/review_eval.py \
      --eval_results data/ui_single_step/eval_results_20260520_123456.json \
      --eval_data data/ui_single_step/eval.json
"""

import argparse
import json
import os
import re
import socket
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_free_port(preferred=8765):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", preferred))
            return preferred
        except OSError:
            s.bind(("", 0))
            return s.getsockname()[1]


class ReviewState:
    def __init__(self, eval_results_path, eval_data_path):
        self.eval_results_path = eval_results_path
        self.eval_data_path = eval_data_path
        self.eval_results = load_json(eval_results_path)
        self.eval_data = load_json(eval_data_path)
        self.errors = [d for d in self.eval_results["details"] if not d["correct"]]
        self.reviewed = set()
        self._build_index()

    def _build_index(self):
        details = self.eval_results.get("details", [])
        self.result_by_id = {d["sample_id"]: d for d in details}

    def reload_eval_data(self):
        self.eval_data = load_json(self.eval_data_path)


STATE: ReviewState | None = None

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Eval Review Tool</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
  .topbar { background: #16213e; padding: 12px 24px; display: flex; align-items: center; gap: 24px;
            border-bottom: 1px solid #0f3460; flex-shrink: 0; }
  .topbar h1 { font-size: 18px; color: #e94560; }
  .topbar .stats { font-size: 14px; color: #a0a0a0; }
  .topbar .stats b { color: #fff; }
  .topbar .nav-btns button { background: #0f3460; color: #fff; border: none; padding: 6px 14px;
                              border-radius: 4px; cursor: pointer; margin-left: 6px; font-size: 13px; }
  .topbar .nav-btns button:hover { background: #e94560; }
  .content { flex: 1; overflow: auto; padding: 20px; }
  .card { background: #16213e; border-radius: 10px; border: 1px solid #0f3460;
          margin-bottom: 16px; overflow: hidden; }
  .card-header { background: #0f3460; padding: 10px 16px; display: flex; justify-content: space-between;
                 align-items: center; font-size: 13px; }
  .card-header .id { color: #e94560; font-weight: bold; }
  .card-header .device { color: #a0a0a0; }
  .card-body { display: flex; min-height: 300px; }
  .col-screenshot { width: 320px; flex-shrink: 0; padding: 12px; display: flex;
                    align-items: center; justify-content: center; background: #111; }
  .col-screenshot img { max-width: 100%; max-height: 400px; border-radius: 4px; }
  .col-actions { flex: 1; display: flex; gap: 2px; }
  .col-gt, .col-pred { flex: 1; padding: 12px; }
  .col-gt { border-right: 2px solid #0f3460; }
  .action-label { font-size: 12px; color: #e94560; text-transform: uppercase; letter-spacing: 1px;
                  margin-bottom: 8px; font-weight: bold; }
  .action-json { font-family: "SF Mono", "Fira Code", monospace; font-size: 13px; line-height: 1.6;
                 white-space: pre-wrap; word-break: break-all; }
  .diff-field { background: rgba(233, 69, 96, 0.2); padding: 1px 4px; border-radius: 3px; }
  .match-field { background: rgba(0, 200, 100, 0.15); padding: 1px 4px; border-radius: 3px; }
  .card-footer { padding: 10px 16px; display: flex; gap: 10px; align-items: center;
                 border-top: 1px solid #0f3460; background: #111827; }
  .btn { padding: 8px 20px; border: none; border-radius: 6px; font-size: 14px; cursor: pointer;
         font-weight: 600; transition: all 0.15s; }
  .btn-accept { background: #00c853; color: #111; }
  .btn-accept:hover { background: #00e676; }
  .btn-reject { background: #ff1744; color: #fff; }
  .btn-reject:hover { background: #ff5252; }
  .btn-skip { background: #424242; color: #fff; }
  .btn-skip:hover { background: #616161; }
  .kbd { display: inline-block; background: #333; color: #fff; padding: 2px 6px; border-radius: 3px;
         font-size: 11px; font-family: monospace; margin-left: 4px; }
  .raw-output { font-family: monospace; font-size: 11px; color: #888; padding: 8px 16px;
                border-top: 1px solid #0f3460; white-space: pre-wrap; word-break: break-all;
                max-height: 80px; overflow: auto; }
  .empty { text-align: center; padding: 60px; color: #666; font-size: 18px; }
  .no-image { color: #555; font-size: 14px; }
</style>
</head>
<body>

<div class="topbar">
  <h1>Eval Review</h1>
  <div class="stats">Errors: <b id="stat-total">0</b> | Reviewed: <b id="stat-reviewed">0</b> | Remaining: <b id="stat-remaining">0</b></div>
  <div class="nav-btns">
    <button onclick="jumpTo(-1)">Prev</button>
    <button onclick="jumpTo(1)">Next</button>
    <button onclick="jumpTo(0)">First Unreviewed</button>
  </div>
</div>

<div class="content" id="content"></div>

<script>
let errors = [];
let currentIndex = 0;

async function loadErrors() {
  const resp = await fetch('/api/errors');
  errors = await resp.json();
  updateStats();
  if (errors.length > 0) {
    renderCard(errors[0]);
  } else {
    document.getElementById('content').innerHTML = '<div class="empty">No error samples to review.</div>';
  }
}

async function updateStats() {
  const resp = await fetch('/api/stats');
  const stats = await resp.json();
  document.getElementById('stat-total').textContent = stats.total;
  document.getElementById('stat-reviewed').textContent = stats.reviewed;
  document.getElementById('stat-remaining').textContent = stats.remaining;
}

function renderCard(item) {
  currentIndex = errors.findIndex(e => e.sample_id === item.sample_id);
  const gt = item.gt_actions || [];
  const pred = item.pred_actions || [];
  const reviewed = item._reviewed;

  const gtHtml = formatActions(gt, pred, 'gt');
  const predHtml = formatActions(pred, gt, 'pred');

  let imgHtml = '<span class="no-image">No image</span>';
  if (item.image_base64) {
    const ext = item.image_base64.startsWith('/9j/') ? 'jpeg' : 'png';
    imgHtml = `<img src="data:image/${ext};base64,${item.image_base64}" alt="screenshot">`;
  }

  const statusBadge = reviewed
    ? '<span style="color:#00c853;font-weight:bold">Reviewed</span>'
    : '<span style="color:#ff9800;font-weight:bold">Pending</span>';

  document.getElementById('content').innerHTML = `
    <div class="card">
      <div class="card-header">
        <span class="id">Sample #${item.sample_id}</span>
        <span>${statusBadge}</span>
        <span class="device">${item.device_type || 'unknown'} | ${currentIndex + 1}/${errors.length}</span>
      </div>
      <div class="card-body">
        <div class="col-screenshot">${imgHtml}</div>
        <div class="col-actions">
          <div class="col-gt">
            <div class="action-label">Ground Truth</div>
            <div class="action-json">${gtHtml}</div>
          </div>
          <div class="col-pred">
            <div class="action-label">Prediction</div>
            <div class="action-json">${predHtml}</div>
          </div>
        </div>
      </div>
      ${item.raw_output ? `<div class="raw-output">${escapeHtml(item.raw_output)}</div>` : ''}
      <div class="card-footer">
        <button class="btn btn-accept" onclick="acceptItem(${item.sample_id})">Accept <span class="kbd">Y</span></button>
        <button class="btn btn-reject" onclick="rejectItem(${item.sample_id})">Reject <span class="kbd">N</span></button>
        <button class="btn btn-skip" onclick="skipItem()">Skip <span class="kbd">S</span></button>
        <span style="margin-left:auto;color:#666;font-size:12px">← → to navigate</span>
      </div>
    </div>
  `;
}

function formatActions(actions, other, side) {
  if (!actions || actions.length === 0) return '<span style="color:#ff1744">PARSE_FAIL</span>';
  return actions.map((a, i) => {
    const fields = Object.entries(a).map(([k, v]) => {
      const otherVal = other && other[i] ? other[i][k] : undefined;
      const cls = (side === 'pred' && otherVal !== undefined && JSON.stringify(v) !== JSON.stringify(otherVal))
        ? 'diff-field' : (side === 'pred' && otherVal !== undefined && JSON.stringify(v) === JSON.stringify(otherVal))
        ? 'match-field' : '';
      return `  <span class="${cls}">"${k}": ${JSON.stringify(v)}</span>`;
    }).join(',\n');
    return `[\n${fields}\n]`;
  }).join(',\n');
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function acceptItem(sampleId) {
  const item = errors[currentIndex];
  const resp = await fetch('/api/accept', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({sample_id: sampleId, pred_actions: item.pred_actions}),
  });
  const result = await resp.json();
  if (result.ok) {
    item._reviewed = true;
    advanceToNext();
  } else {
    alert('Error: ' + result.error);
  }
}

function rejectItem(sampleId) {
  errors[currentIndex]._reviewed = true;
  advanceToNext();
}

function skipItem() {
  advanceToNext();
}

function advanceToNext() {
  updateStats();
  for (let i = 1; i <= errors.length; i++) {
    const idx = (currentIndex + i) % errors.length;
    if (!errors[idx]._reviewed) {
      renderCard(errors[idx]);
      return;
    }
  }
  document.getElementById('content').innerHTML = '<div class="empty">All samples reviewed!</div>';
  updateStats();
}

function jumpTo(delta) {
  if (delta === 0) {
    for (let i = 0; i < errors.length; i++) {
      if (!errors[i]._reviewed) { renderCard(errors[i]); return; }
    }
    return;
  }
  const next = (currentIndex + delta + errors.length) % errors.length;
  renderCard(errors[next]);
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const item = errors[currentIndex];
  if (!item) return;
  if (e.key === 'y' || e.key === 'Y') { acceptItem(item.sample_id); e.preventDefault(); }
  else if (e.key === 'n' || e.key === 'N') { rejectItem(item.sample_id); e.preventDefault(); }
  else if (e.key === 's' || e.key === 'S') { skipItem(); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { jumpTo(-1); e.preventDefault(); }
  else if (e.key === 'ArrowRight') { jumpTo(1); e.preventDefault(); }
});

loadErrors();
</script>
</body>
</html>"""


class ReviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._send_html(HTML_PAGE)
        elif self.path == "/api/errors":
            self._api_errors()
        elif self.path == "/api/stats":
            self._api_stats()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/accept":
            self._api_accept()
        else:
            self.send_error(404)

    def _send_html(self, html):
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json_response(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _api_errors(self):
        result = []
        for err in STATE.errors:
            sid = err["sample_id"]
            eval_sample = STATE.eval_data[sid] if sid < len(STATE.eval_data) else None
            item = {
                "sample_id": sid,
                "device_type": err.get("device_type"),
                "gt_actions": err.get("gt_actions"),
                "pred_actions": err.get("pred_actions"),
                "raw_output": err.get("raw_output", ""),
                "new_tokens": err.get("new_tokens"),
                "_reviewed": sid in STATE.reviewed,
            }
            if eval_sample:
                item["image_base64"] = eval_sample.get("image_base64", "")
            result.append(item)
        self._json_response(result)

    def _api_stats(self):
        self._json_response({
            "total": len(STATE.errors),
            "reviewed": len(STATE.reviewed),
            "remaining": len(STATE.errors) - len(STATE.reviewed),
        })

    def _api_accept(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._json_response({"ok": False, "error": "Invalid JSON"}, 400)
            return

        sample_id = req.get("sample_id")
        pred_actions = req.get("pred_actions")
        if sample_id is None or pred_actions is None:
            self._json_response({"ok": False, "error": "Missing sample_id or pred_actions"}, 400)
            return

        if sample_id >= len(STATE.eval_data):
            self._json_response({"ok": False, "error": "sample_id out of range"}, 400)
            return

        sample = STATE.eval_data[sample_id]

        if "accepted_actions" not in sample:
            gt_actions = _extract_gt_actions(sample)
            sample["accepted_actions"] = [gt_actions] if gt_actions else []

        sample["accepted_actions"].append(pred_actions)

        save_json(STATE.eval_data_path, STATE.eval_data)
        STATE.reviewed.add(sample_id)

        print(f"  Accepted sample #{sample_id}: {json.dumps(pred_actions, ensure_ascii=False)[:80]}")
        self._json_response({"ok": True})

    def log_message(self, format, *args):
        pass


def _extract_gt_actions(sample):
    messages = sample.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            try:
                obj = json.loads(msg["content"])
                return obj.get("actions", [])
            except (json.JSONDecodeError, KeyError):
                pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Eval Review Tool")
    parser.add_argument("--eval_results", required=True, help="Path to eval_results_*.json")
    parser.add_argument("--eval_data", required=True, help="Path to eval.json")
    parser.add_argument("--port", type=int, default=0, help="Port (default: auto)")
    args = parser.parse_args()

    eval_results_path = Path(args.eval_results)
    eval_data_path = Path(args.eval_data)
    if not eval_results_path.exists():
        print(f"Error: {eval_results_path} not found", file=sys.stderr)
        sys.exit(1)
    if not eval_data_path.exists():
        print(f"Error: {eval_data_path} not found", file=sys.stderr)
        sys.exit(1)

    global STATE
    STATE = ReviewState(str(eval_results_path), str(eval_data_path))

    port = args.port or find_free_port()
    server = HTTPServer(("0.0.0.0", port), ReviewHandler)

    print(f"Evaluation Review Tool")
    print(f"  Eval results: {eval_results_path}")
    print(f"  Eval data:    {eval_data_path}")
    print(f"  Error samples: {len(STATE.errors)}")
    print(f"\n  Open http://localhost:{port} in your browser")
    print(f"  Keyboard: Y=Accept, N=Reject, S=Skip, ←→=Navigate\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
