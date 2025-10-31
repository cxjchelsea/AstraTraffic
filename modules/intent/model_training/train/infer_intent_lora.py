# -*- coding: utf-8 -*-
"""
一键评测：BERT + LoRA（仅主意图，单标签CE）意图识别（测试集）
- 评估：accuracy、macro F1、逐类 P/R/F1（显式 labels 保证不越界）
- 导出 Excel：text / true_label / pred_label / correct / 各类概率(prob_*)
"""

import os
import json
import re
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import pandas as pd  # 导出 Excel

# =========================
# 0) 固定配置（按需修改）
# =========================
BASE_MODEL_PATH = r"../models/bert-base-chinese"  # 底座 BERT
ADAPTER_DIR     = r"../models/bert-intent-lora-v1/ds_single"  # LoRA 目录
HEAD_BIN        = os.path.join(ADAPTER_DIR, "head.bin")         # 主头权重
HEAD_CFG        = os.path.join(ADAPTER_DIR, "head_config.json") # 主头配置
LABEL_MAP_PATH  = r"../data/intent/label_map_intent.json"       # 10 类映射（两种格式均可）
TEST_PATH       = r"../data/intent/ds_fixed_async_test.json"    # 测试集
MAX_LEN         = 256
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# 导出 Excel 文件名
EXPORT_XLSX     = os.path.join(ADAPTER_DIR, "primary_eval.xlsx")

# 标签别名（数据里的别名 → label_map 标准名）
ALIAS = {
    "系统功能咨询": "系统操作",
    "系统设置":   "系统操作",
    "功能咨询":   "系统操作",
    # 如无需此条，可删
    "环境健康":   "日常生活",
}

# =========================
# 1) 工具函数（仅主意图）
# =========================
def load_label_map(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        m = json.load(f)
    if isinstance(m, dict) and "label2id" in m and "id2label" in m:
        label2id = {k: int(v) for k, v in m["label2id"].items()}
        id2label = {int(k): v for k, v in m["id2label"].items()}
    elif all(isinstance(v, int) for v in m.values()):
        label2id = {k: int(v) for k, v in m.items()}
        id2label = {v: k for k, v in label2id.items()}
    else:
        id2label = {int(k): v for k, v in m.items()}
        label2id = {v: k for k, v in id2label.items()}
    id2label = {i: id2label[i] for i in sorted(id2label.keys())}
    label2id = {v: k for k, v in id2label.items()}
    return label2id, id2label

def normalize_label(lbl: str, label2id: Dict[str, int]) -> str:
    if lbl in label2id:
        return lbl
    if lbl in ALIAS and ALIAS[lbl] in label2id:
        return ALIAS[lbl]
    return None

def load_any_json(fp: str):
    if not os.path.exists(fp):
        print(f"[WARN] 文件不存在：{fp}")
        return []
    with open(fp, "r", encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            return json.load(f)  # JSON 数组
        return [json.loads(line) for line in f if line.strip()]  # JSONL

def normalize_items_primary_only(items):
    out = []
    for x in items:
        text = x.get("text")
        if isinstance(x.get("data"), dict):
            text = text or x["data"].get("text")
        if not text:
            continue

        primary = None
        anns = x.get("annotations", [])
        if anns and isinstance(anns, list):
            for r in anns[0].get("result", []):
                if not isinstance(r, dict):
                    continue
                if r.get("from_name") != "primary":
                    continue
                val = r.get("value", {})
                if not isinstance(val, dict):
                    continue
                ch = val.get("choices", [])
                if ch:
                    primary = str(ch[0]).strip()
                    break

        if not primary:
            data = x.get("data") or {}
            wp = data.get("weak_primary")
            if wp:
                primary = str(wp).strip()

        if primary:
            out.append({"text": str(text).strip(), "primary_intent": primary})
    return out

def to_primary_gold(items, label2id):
    texts, y_primary_true, true_primary_names = [], [], []
    dropped = 0
    for x in items:
        if "text" not in x:
            continue
        p = x.get("primary_intent")
        p_norm = normalize_label(p, label2id) if p else None
        if p_norm is None:
            dropped += 1
            continue
        texts.append(x["text"])
        y_primary_true.append(label2id[p_norm])
        true_primary_names.append(p_norm)
    if dropped > 0:
        print(f"[WARN] 丢弃无主意图或主意图未知样本 {dropped} 条。")
    return texts, np.array(y_primary_true, dtype=np.int64), true_primary_names

# =========================
# 2) 只有“主头”的推理模块
# =========================
class PrimaryHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, pooled):
        x = self.dropout(pooled)
        return self.classifier(x)

@torch.inference_mode()
def predict_primary_probs(texts, tokenizer, encoder, head, max_len=256, device="cpu"):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = encoder.base_model(**enc)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        pooled = outputs.last_hidden_state[:, 0]
    logits = head(pooled)
    return torch.softmax(logits, dim=-1).cpu().numpy()

# =========================
# 3) 主流程
# =========================
def main():
    print(f"[INFO] 加载标签映射：{os.path.abspath(LABEL_MAP_PATH)}")
    label2id, id2label = load_label_map(LABEL_MAP_PATH)
    num_labels = len(id2label)
    label_names = [id2label[i] for i in range(num_labels)]
    print(f"[INFO] 类别数：{num_labels} → {label_names}")

    print(f"[INFO] 读取测试集：{os.path.abspath(TEST_PATH)}")
    raw = load_any_json(TEST_PATH)
    data = normalize_items_primary_only(raw)

    texts, y_true, true_names = to_primary_gold(data, label2id)
    print(f"[INFO] 测试集有效样本数：{len(texts)}")

    print("[INFO] 加载分词器与模型 ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
    base = AutoModel.from_pretrained(BASE_MODEL_PATH)
    encoder = PeftModel.from_pretrained(base, ADAPTER_DIR).to(DEVICE).eval()

    with open(HEAD_CFG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    hidden = cfg.get("hidden_size", base.config.hidden_size)
    head = PrimaryHead(hidden, num_labels)
    state = torch.load(HEAD_BIN, map_location="cpu")
    sd = state.get("primary_head", state)  # 兼容两种保存形式
    head.classifier.load_state_dict(sd)
    head.to(DEVICE).eval()
    print(f"[INFO] 模型与适配器已加载：device={DEVICE}")

    print("[INFO] 推理中 ...")
    probs = predict_primary_probs(texts, tokenizer, encoder, head, MAX_LEN, DEVICE)
    y_pred = np.argmax(probs, axis=1)

    primary_acc = accuracy_score(y_true, y_pred)
    primary_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print("\n[TEST] Primary (主意图) metrics")
    print({"primary_acc": round(primary_acc, 4), "primary_macro_f1": round(primary_macro_f1, 4)})

    all_labels = list(range(num_labels))
    p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average=None, zero_division=0
    )
    print("\n[TEST] Primary per-class P/R/F1")
    for i, name in enumerate(label_names):
        print(f"{name: <8}  P={p_c[i]:.3f}  R={r_c[i]:.3f}  F1={f1_c[i]:.3f}  supp={int(sup_c[i])}")

    # ========= 导出 Excel =========
    print(f"\n[INFO] 正在导出 Excel 到：{EXPORT_XLSX}")
    records = []
    for i, text in enumerate(texts):
        row = {
            "text": text,
            "true_label": id2label[int(y_true[i])],
            "pred_label": id2label[int(y_pred[i])],
            "correct": bool(y_true[i] == y_pred[i]),
        }
        # 各类概率展开为列
        for j, name in enumerate(label_names):
            row[f"prob_{name}"] = float(probs[i, j])
        records.append(row)

    df = pd.DataFrame(records, columns=["text", "true_label", "pred_label", "correct"] + [f"prob_{n}" for n in label_names])
    # 为避免过长文本被截断，建议关闭 index，保留列名
    with pd.ExcelWriter(EXPORT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="primary_eval", index=False)
        # 另存逐类指标
        df_cls = pd.DataFrame({
            "label": label_names,
            "precision": p_c,
            "recall": r_c,
            "f1": f1_c,
            "support": sup_c.astype(int),
        })
        df_cls.to_excel(writer, sheet_name="per_class_metrics", index=False)
        # 总体指标
        df_overall = pd.DataFrame([{
            "primary_acc": primary_acc,
            "primary_macro_f1": primary_macro_f1,
            "num_samples": len(texts)
        }])
        df_overall.to_excel(writer, sheet_name="overall", index=False)

    print("[INFO] 导出完成。")

if __name__ == "__main__":
    main()
