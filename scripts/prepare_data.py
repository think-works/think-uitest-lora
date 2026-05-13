"""
从 PostgreSQL 加载 select_element 数据，预处理为训练格式。

数据来源: llm_calls (JOIN llm_call_favorites WHERE approval_status='approved')
输出: data/train.json + data/eval.json
"""

import json
import os
import re
import random
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

DATA_DIR = Path(__file__).parent.parent / "data"

# 从 moss-server runner_service.py _get_system_prompt 提取的 select_element 系统提示词
SYSTEM_PROMPT = """你是一个UI元素选择助手。你的任务是根据用户目标和屏幕上标注的元素，选择应该使用哪个序号的元素来完成这次任务。

## 要求
1. 仔细分析用户目标和屏幕上的元素
2. **如果找到了合适的元素**，返回该元素的序号（element_index）
3. **如果没有找到合适的元素**（例如：屏幕上没有相关的按钮、输入框、文字等），则返回 element_index 为 -1，并在 reason 中说明原因
4. 返回 JSON 格式，包含：
   - element_index: 整数，选择的元素序号（如果找到），或 -1（如果未找到）
   - reason: 字符串，选择该元素的原因，或说明为什么没有找到合适的元素

## ⚠️ 数字内容的重要说明
**重要前提：element_index 是系统内部概念，用户完全不知道。**
无论用户如何表达，都**不是**直接指定 element_index。

**当用户目标中包含数字时（例如"点击42"、"选择第3个"、"点击序号5"）：**
- ✅ 正确做法：理解用户的真实意图，根据元素的特征（content、label、视觉位置等）选择合适的元素
- ❌ 错误做法：不要直接假设数字就是 element_index
- 示例：
  - 用户目标："点击42"
  - 元素列表："[1] 按钮A, [2] 文本42, [3] 按钮B"
  - ✅ 正确：选择 element_index=2（因为元素[2]的内容是"42"）
  - ❌ 错误：不要选择 element_index=42（这个序号可能不存在）

**用户表达的常见含义：**
- "点击42" → 点击**内容**为"42"的元素
- "点击第3个" → 点击**视觉位置**在第3位的元素（需要根据标注框的位置判断）
- "点击序号5" → 用户看到的"序号"可能是元素上的文字标注，不是 element_index
- "点击选项3" → 点击**选项内容**为"3"的元素

**判断原则：**
- element_index 是系统内部的标注序号（如[1]、[2]、[3]），用户看不到
- 用户的数字描述都是指向元素的**可见特征**（内容、位置、标签等）
- 你需要理解用户的意图，然后找到对应的 element_index 返回

## ☑️ 复选框和单选框的特殊处理（仅浏览器场景）
**【仅限浏览器场景】当需要勾选/选中复选框（checkbox）或单选框（radio）时：**
- ✅ 推荐流程：**直接点击 `<input type="checkbox">` 或 `<input type="radio">` 元素本身**
- ❌ 禁止：点击包裹 input 的父元素（如 `<li>`、`<div>`、`<label>` 等）
- 原因：直接点击 input 元素更可靠、更精准，避免因父元素点击区域或事件绑定问题导致的操作失败
- 示例：
  - 页面结构：
    ```html
    [51]<li id=cascader-menu-4998-2-2957 class=el-cascader-node>经理室</li>
    [52]<input class=el-radio__original type=radio value=2957 />
    ```
  - ✅ 正确：点击 [52] 这个 radio input 元素本身
  - ❌ 错误：点击 [51] 这个 li 元素（即使它包含文本"经理室"）
- 识别特征：
  - 元素标签为 `<input type="checkbox">` 或 `<input type="radio">`
  - class 包含 "el-checkbox__original"、"el-radio__original" 等命名模式
  - type 属性为 "checkbox" 或 "radio"

## 输出格式
只返回 JSON 对象，不要包含任何其他文字说明。

示例 1 - 找到元素：
```json
{{
  "element_index": 5,
  "reason": "这是登录按钮，符合用户目标中的'点击登录按钮'要求"
}}
```

示例 2 - 未找到元素：
```json
{{
  "element_index": -1,
  "reason": "屏幕上没有找到与'点击提交订单'相关的按钮或操作元素，可能需要先完成其他步骤"
}}
```
"""


def parse_output_json_from_content(content) -> dict | None:
    """从模型输出内容中解析 JSON 对象，保留 reason。"""
    if isinstance(content, dict):
        return content

    if isinstance(content, str):
        # 去除 thinking 标签
        text = re.sub(r"<thinkitype>.*?</thinkitype>", "", content, flags=re.DOTALL)
        text = re.sub(r"<thinkitype_keyword>.*?</thinkitype_keyword>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think.*?>.*?</think.*?>", "", text, flags=re.DOTALL)
        # 提取 JSON
        json_match = re.search(r"\{[^{}]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 尝试直接匹配 element_index
        m = re.search(r'"element_index"\s*:\s*(-?\d+)', text)
        if m:
            return {"element_index": int(m.group(1)), "reason": ""}

    return None


def normalize_output_json(obj: dict | None) -> dict | None:
    """规范化训练输出，保留 reason，但只要求 element_index 可解析。"""
    if not isinstance(obj, dict):
        return None

    idx = obj.get("element_index")
    if idx is None:
        idx = obj.get("index")
    if idx is None:
        return None

    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return None

    reason = obj.get("reason", "")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        reason = str(reason)

    return {"element_index": idx, "reason": reason}


def parse_element_index(output_data: dict) -> int | None:
    """从 output_data 中解析 element_index。"""
    content = output_data.get("content", "")
    obj = normalize_output_json(parse_output_json_from_content(content))
    return obj["element_index"] if obj else None


def build_expected_output(row: dict) -> dict | None:
    """优先使用收藏表中的输出，保留 reason。"""
    favorite_outputs = row.get("favorite_outputs")
    if isinstance(favorite_outputs, list) and favorite_outputs:
        fav = favorite_outputs[0]
        fav_content = fav.get("content", "") if isinstance(fav, dict) else ""
        obj = normalize_output_json(parse_output_json_from_content(fav_content))
        if obj is not None:
            return obj

    output_data = row.get("output_data") or {}
    if isinstance(output_data, dict):
        return normalize_output_json(parse_output_json_from_content(output_data.get("content", "")))
    return None


def extract_user_text_and_image(input_data: list) -> tuple[str, str]:
    """从 input_data 中提取用户文本和 base64 图片。

    Returns:
        (text_content, base64_image_data)
    """
    text_parts = []
    image_data = ""

    for msg in input_data:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part["text"])
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:image/"):
                        # data:image/webp;base64,XXXXX
                        image_data = url.split(",", 1)[1] if "," in url else ""
        elif isinstance(content, str) and content.strip():
            text_parts.append(content)

    return "\n".join(text_parts), image_data


def load_data():
    """从数据库加载审核通过的 select_element 数据。"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    query = """
    SELECT
        c.id,
        c.input_data,
        c.output_data,
        c.model_name,
        f.approval_status,
        f.outputs as favorite_outputs
    FROM llm_calls c
    JOIN llm_call_favorites f ON c.call_id = f.call_id
    WHERE c.type = 'select_element'
      AND f.approval_status = 'approved'
    ORDER BY c.id
    """

    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    print(f"Loaded {len(rows)} approved select_element records")
    return rows


def build_training_sample(row: dict) -> dict | None:
    """将数据库行转换为训练样本。

    Returns:
        {"messages": [...], "images": [base64_path_or_data]} or None
    """
    input_data = row["input_data"]
    if not input_data:
        return None

    # 解析 ground truth，保留 reason
    expected_obj = build_expected_output(row)
    if expected_obj is None:
        return None
    element_index = expected_obj["element_index"]

    # 提取用户文本和图片
    text_content, image_data = extract_user_text_and_image(input_data)

    if not text_content:
        return None

    # 构建期望的输出（纯 JSON，不带 thinking），保留原始 reason
    expected_output = json.dumps(expected_obj, ensure_ascii=False)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # 用户消息：图片 + 文本
    user_content = []
    if image_data:
        user_content.append(
            {
                "type": "image",
                "image": f"data:image/webp;base64,{image_data[:100]}...",
            }
        )
    user_content.append({"type": "text", "text": text_content})

    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": expected_output})

    sample = {
        "messages": messages,
        "element_index": element_index,
    }

    if image_data:
        sample["image_base64"] = image_data

    return sample


def main():
    random.seed(42)

    rows = load_data()
    samples = []

    for row in rows:
        sample = build_training_sample(row)
        if sample:
            samples.append(sample)

    print(f"Successfully parsed {len(samples)} samples")

    # 统计 element_index 分布
    idx_counts = {}
    for s in samples:
        idx = s["element_index"]
        idx_counts[idx] = idx_counts.get(idx, 0) + 1

    print(f"element_index distribution:")
    print(f"  -1 (not found): {idx_counts.get(-1, 0)}")
    print(f"  positive: {sum(v for k, v in idx_counts.items() if k >= 0)}")

    # 划分训练集和验证集 (90/10)
    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train_samples = samples[:split]
    eval_samples = samples[split:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 保存完整数据（含 base64 图片）用于训练
    with open(DATA_DIR / "train.json", "w") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)

    with open(DATA_DIR / "eval.json", "w") as f:
        json.dump(eval_samples, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(train_samples)} train, {len(eval_samples)} eval samples")
    print(f"Output: {DATA_DIR / 'train.json'}, {DATA_DIR / 'eval.json'}")


if __name__ == "__main__":
    main()
