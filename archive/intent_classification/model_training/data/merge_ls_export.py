#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 Label Studio 导出的 JSON 合并为训练集 JSONL：
- 主意图：必选（未标则报错）
- 次意图：可为空（空则输出 []）
输出字段：
  text, primary_intent, secondary_intents
"""

import json
from typing import List, Dict, Any

IN_EXPORT_JSON = "label_studio_export.json"   # 你的 LS 导出文件
OUT_TRAIN_JSONL = "intent_labeled.jsonl"

def parse_one_result(res: Dict[str, Any]) -> Dict[str, Any]:
    # 找出 primary 与 secondary
    primary = None
    secondary: List[str] = []
    for r in res.get("result", []):
        if r.get("type") == "choices" and r.get("from_name") == "primary":
            # choices 是单选
            ch = r.get("value", {}).get("choices", [])
            if ch:
                primary = ch[0]
        if r.get("type") == "choices" and r.get("from_name") == "secondary":
            ch = r.get("value", {}).get("choices", [])
            if ch:
                secondary = list(dict.fromkeys(ch))  # 去重保序
    return {"primary": primary, "secondary": secondary}

def main():
    print(f"📥 Loading: {IN_EXPORT_JSON}")
    data = json.load(open(IN_EXPORT_JSON, "r", encoding="utf-8"))
    out = open(OUT_TRAIN_JSONL, "w", encoding="utf-8")

    n_ok = 0
    for task in data:
        text = (task.get("data") or {}).get("text", "").strip()
        if not text:
            continue
        # 可能有多个标注，取最后一个已完成的或第一个
        annos = task.get("annotations") or []
        if not annos:
            continue
        ann = None
        # 优先 chosen last_finished
        for a in annos:
            if a.get("ground_truth") or a.get("was_cancelled"):
                continue
            ann = a
        if ann is None:
            ann = annos[0]

        parsed = parse_one_result(ann)
        primary = parsed["primary"]
        secondary = parsed["secondary"]

        if not primary:
            # 违反“主必选”规则，直接跳过或抛错
            raise ValueError(f"样本缺少主意图：{text[:50]}...")

        rec = {"text": text, "primary_intent": primary, "secondary_intents": secondary}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_ok += 1

    out.close()
    print(f"✅ 合并完成：{OUT_TRAIN_JSONL}  样本数={n_ok}")

if __name__ == "__main__":
    main()
