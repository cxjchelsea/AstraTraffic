# -*- coding: utf-8 -*-
"""
一键评测：BERT + LoRA（多任务：主=CE、次=BCE）意图识别（测试集）
- 无需命令行参数；路径与阈值在常量里配置
- 兼容 Label Studio / JSON / JSONL
- 评估：
  * 主意图：accuracy、macro F1
  * 次意图：micro/macro F1 + 每类 P/R/F1
- 导出 CSV：文本、真值主/次、预测主/次、各类概率（主头/次头）
"""

import os
import json
import csv
import re
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# =========================
# 0) 固定配置（按需修改）
# =========================
BASE_MODEL_PATH = r"../models/bert-base-chinese"  # 底座 BERT
ADAPTER_DIR     = r"../models/bert-intent-lora-v1/ds_multitask"  # 你的 LoRA 多任务适配器目录
HEADS_BIN       = os.path.join(ADAPTER_DIR, "heads.bin")       # 两个头的权重
HEADS_CFG       = os.path.join(ADAPTER_DIR, "heads_config.json")
LABEL_MAP_PATH  = r"../data/intent/label_map_infer.json"  # 10 类映射（两种格式均可）
TEST_PATH       = r"../data/intent/ds_fixed_async_test.json"  # 测试集
MAX_LEN         = 256

# 多标签阈值（用于“次意图”判定）
THRESHOLD       = 0.50

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# 标签别名（数据里的别名 → label_map 标准名）
ALIAS = {
    "系统功能咨询": "系统操作",
    "系统设置":   "系统操作",
    "功能咨询":   "系统操作",
    "环境健康":   "日常生活",
}

# 一些不希望作为“次意图”输出的标签（可选）
BLOCK_SECONDARY = {"闲聊其他"}

# =========================
# 1) 工具函数
# =========================
def load_label_map(fp: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(fp, "r", encoding="utf-8") as f:
        m = json.load(f)
    if all(isinstance(v, int) for v in m.values()):
        label2id = {k: int(v) for k, v in m.items()}
        id2label = {v: k for k, v in label2id.items()}
    else:
        id2label = {int(k): v for k, v in m.items()}
        id2label = {i: id2label[i] for i in sorted(id2label)}
        label2id = {v: k for k, v in id2label.items()}
    return label2id, id2label

def normalize_label(lbl: str, label2id: Dict[str, int]) -> str:
    if lbl in label2id:
        return lbl
    if lbl in ALIAS and ALIAS[lbl] in label2id:
        return ALIAS[lbl]
    return None

def load_any_json(fp: str) -> List[Dict[str, Any]]:
    if not os.path.exists(fp):
        print(f"[WARN] 文件不存在：{fp}")
        return []
    with open(fp, "r", encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            return json.load(f)  # JSON 数组
        return [json.loads(line) for line in f if line.strip()]  # JSONL

def normalize_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    统一输出：
      {"text": "...", "primary_intent": "...", "secondary_intents": ["..."] }
    - 强标注：annotations.primary / annotations.secondary
    - 兜底：data.weak_primary / data.weak_secondary（多分隔符）
    """
    out = []
    for x in items:
        text = x.get("text")
        if isinstance(x.get("data"), dict):
            text = text or x["data"].get("text")
        if not text:
            continue

        primary = None
        secondaries = []

        anns = x.get("annotations", [])
        if anns and isinstance(anns, list):
            for r in anns[0].get("result", []):
                if not isinstance(r, dict): continue
                val = r.get("value", {})
                if not isinstance(val, dict): continue
                ch = val.get("choices", [])
                if not ch: continue
                if r.get("from_name") == "primary" and primary is None:
                    primary = str(ch[0]).strip()
                elif r.get("from_name") == "secondary":
                    for c in ch:
                        c = str(c).strip()
                        if c: secondaries.append(c)

        # 兜底 weak_*
        data = x.get("data") or {}
        if not primary:
            wp = data.get("weak_primary")
            if wp:
                primary = str(wp).strip()
        ws = data.get("weak_secondary")
        if ws:
            for c in re.split(r"[,\u3001\uFF0C;\uFF1B/\s]+", str(ws)):
                c = c.strip()
                if c: secondaries.append(c)

        rec = {"text": str(text).strip()}
        if primary:
            rec["primary_intent"] = primary
        if secondaries:
            rec["secondary_intents"] = sorted(set(secondaries))
        out.append(rec)
    return out

def to_multitask_gold(items: List[Dict[str, Any]], label2id: Dict[str, int]):
    """
    返回：
      texts: List[str]
      y_primary_true: np.ndarray shape=(N,)  单标签id
      Y_secondary_true: np.ndarray shape=(N, K) 多标签multi-hot
      true_primary_names: List[str]
      true_secondary_names: List[List[str]]
    约束：主意图必须有，次意图可0~N
    """
    K = len(label2id)
    texts = []
    y_primary_true = []
    Y_secondary_true = []
    true_primary_names = []
    true_secondary_names = []
    dropped = 0

    for x in items:
        if "text" not in x: continue

        # 主
        p = x.get("primary_intent")
        p_norm = normalize_label(p, label2id) if p else None
        if p_norm is None:
            dropped += 1
            continue
        pid = label2id[p_norm]

        # 次
        multi = [0] * K
        sec_names = []
        for s in x.get("secondary_intents", []) or []:
            s_norm = normalize_label(s, label2id)
            if s_norm:
                multi[label2id[s_norm]] = 1
                sec_names.append(s_norm)

        texts.append(x["text"])
        y_primary_true.append(pid)
        Y_secondary_true.append(multi)
        true_primary_names.append(p_norm)
        true_secondary_names.append(sorted(set(sec_names)))

    if dropped > 0:
        print(f"[WARN] 丢弃无主意图或主意图未知样本 {dropped} 条。")

    return (
        texts,
        np.array(y_primary_true, dtype=np.int64),
        np.array(Y_secondary_true, dtype=np.int64),
        true_primary_names,
        true_secondary_names
    )

# =========================
# 2) 两个头（推理）
# =========================
class MultiTaskHeads(nn.Module):
    """两个头：主（CE），次（BCE）"""
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.primary_head = nn.Linear(hidden_size, num_labels)
        self.secondary_head = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled):
        x = self.dropout(pooled)
        logits_p = self.primary_head(x)    # [N, K]
        logits_s = self.secondary_head(x)  # [N, K]
        return logits_p, logits_s

@torch.inference_mode()
def predict_probs(texts: List[str], tokenizer, encoder, heads, max_len=256, device="cpu"):
    enc = tokenizer(
        texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    # 取 encoder 的 base_model 输出（LoRA 已注入 encoder）
    outputs = encoder.base_model(**enc)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        pooled = outputs.last_hidden_state[:, 0]

    logits_p, logits_s = heads(pooled)               # [N,K], [N,K]
    probs_p = torch.softmax(logits_p, dim=-1).cpu().numpy()
    probs_s = torch.sigmoid(logits_s).cpu().numpy()
    return probs_p, probs_s

# =========================
# 3) 主流程
# =========================
def main():
    print(f"[INFO] 加载标签映射：{os.path.abspath(LABEL_MAP_PATH)}")
    label2id, id2label = load_label_map(LABEL_MAP_PATH)
    num_labels = len(id2label)
    label_names = [id2label[i] for i in range(num_labels)]
    print(f"[INFO] 类别数：{num_labels} → {label_names}")

    # 载入测试集并标准化
    print(f"[INFO] 读取测试集：{os.path.abspath(TEST_PATH)}")
    raw = load_any_json(TEST_PATH)
    data = normalize_items(raw)

    # 组装主/次真值
    (
        texts,
        y_primary_true,
        Y_secondary_true,
        true_primary_names,
        true_secondary_names
    ) = to_multitask_gold(data, label2id)
    print(f"[INFO] 测试集有效样本数：{len(texts)}")

    # 载入模型：分词器、底座、LoRA、heads
    print("[INFO] 加载分词器与模型 ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
    base = AutoModel.from_pretrained(BASE_MODEL_PATH)
    encoder = PeftModel.from_pretrained(base, ADAPTER_DIR).to(DEVICE).eval()

    with open(HEADS_CFG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    hidden = cfg.get("hidden_size", base.config.hidden_size)
    heads = MultiTaskHeads(hidden, num_labels)
    state = torch.load(HEADS_BIN, map_location="cpu")
    heads.primary_head.load_state_dict(state["primary_head"])
    heads.secondary_head.load_state_dict(state["secondary_head"])
    heads.to(DEVICE).eval()
    print(f"[INFO] 模型与适配器已加载：device={DEVICE}")

    # 推理
    print("[INFO] 推理中 ...")
    probs_p, probs_s = predict_probs(texts, tokenizer, encoder, heads, MAX_LEN, DEVICE)
    # 主：取 argmax
    y_primary_pred = np.argmax(probs_p, axis=1)
    # 次：阈值判定（可屏蔽某些类）
    Y_secondary_pred = (probs_s >= THRESHOLD).astype(int)
    # 不把“主意图”再算进“次意图”
    for i in range(len(texts)):
        Y_secondary_pred[i, y_primary_pred[i]] = 0
        # 黑名单过滤
        for j, name in enumerate(label_names):
            if name in BLOCK_SECONDARY:
                Y_secondary_pred[i, j] = 0

    # 评估 —— 主意图
    primary_acc = accuracy_score(y_primary_true, y_primary_pred)
    primary_macro_f1 = f1_score(y_primary_true, y_primary_pred, average="macro", zero_division=0)
    print("\n[TEST] Primary (主意图) metrics")
    print({"primary_acc": round(primary_acc, 4), "primary_macro_f1": round(primary_macro_f1, 4)})

    # —— 新增：主意图逐类 P/R/F1 & 支持度 ——
    p_c_p, r_c_p, f1_c_p, sup_c_p = precision_recall_fscore_support(
        y_primary_true, y_primary_pred, average=None, zero_division=0
    )
    print("\n[TEST] Primary per-class P/R/F1")
    for i, name in enumerate(label_names):
        print(f"{name: <8}  P={p_c_p[i]:.3f}  R={r_c_p[i]:.3f}  F1={f1_c_p[i]:.3f}  supp={int(sup_c_p[i])}")

    # 评估 —— 次意图
    sec_micro_f1 = f1_score(Y_secondary_true, Y_secondary_pred, average="micro", zero_division=0)
    sec_macro_f1 = f1_score(Y_secondary_true, Y_secondary_pred, average="macro", zero_division=0)
    print("\n[TEST] Secondary (次意图) metrics")
    print({"secondary_micro_f1": round(sec_micro_f1, 4), "secondary_macro_f1": round(sec_macro_f1, 4)})

    p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(
        Y_secondary_true, Y_secondary_pred, average=None, zero_division=0
    )
    print("\n[TEST] Secondary per-class P/R/F1")
    for i, name in enumerate(label_names):
        print(f"{name: <8}  P={p_c[i]:.3f}  R={r_c[i]:.3f}  F1={f1_c[i]:.3f}  supp={int(sup_c[i])}")


if __name__ == "__main__":
    main()
