# -*- coding: utf-8 -*-
"""
交互推理：BERT + LoRA（仅主意图）
- 加载：BERT底座 + LoRA适配器 + 主头（head.bin / head_config.json）
- 交互输入中文，输出：
  * 主意图（唯一、置信度）
  * TopK 概览（来自主头）
"""

import os
import json
from typing import List, Dict

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# =======================
# 0) 固定配置（按需修改）
# =======================
from settings import (
    INTENT_BASE_MODEL_PATH, INTENT_ADAPTER_DIR, LABEL_MAP_PATH
)
BASE_MODEL_PATH = INTENT_BASE_MODEL_PATH
ADAPTER_DIR     = INTENT_ADAPTER_DIR
HEAD_BIN        = os.path.join(ADAPTER_DIR, "head.bin")
HEAD_CFG        = os.path.join(ADAPTER_DIR, "head_config.json")
LABEL_MAP_FILE  = LABEL_MAP_PATH  # 可能不存在，load 时判断
MAX_LEN         = 256
TOPK            = 5

# =======================
# 1) 工具函数
# =======================
def load_label_map():
    if os.path.exists(LABEL_MAP_FILE):
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
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

    with open(HEAD_CFG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    id2label = {int(k): v for k, v in cfg.get("id2label", {}).items()}
    id2label = {i: id2label[i] for i in sorted(id2label.keys())}
    label2id = {v: k for k, v in id2label.items()}
    return label2id, id2label

# =======================
# 2) 模型（仅主头）
# =======================
class PrimaryHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, pooled):
        x = self.dropout(pooled)
        return self.classifier(x)

class PrimaryInfer:
    def __init__(self):
        # 标签
        self.label2id, self.id2label = load_label_map()
        self.labels = [self.id2label[i] for i in range(len(self.id2label))]
        self.num_labels = len(self.labels)

        # 分词器
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)

        # 底座 + LoRA
        base = AutoModel.from_pretrained(BASE_MODEL_PATH)
        self.encoder = PeftModel.from_pretrained(base, ADAPTER_DIR)

        # 读取 head_config
        with open(HEAD_CFG, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        hidden = cfg.get("hidden_size", base.config.hidden_size)

        # 构造 & 加载主头
        self.head = PrimaryHead(hidden, self.num_labels)
        state = torch.load(HEAD_BIN, map_location="cpu")
        sd = state.get("primary_head", state)  # 兼容两种保存形式
        self.head.classifier.load_state_dict(sd)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device).eval()
        self.head.to(self.device).eval()

        print(f"[INFO] Loaded encoder+LoRA from {ADAPTER_DIR}")
        print(f"[INFO] Loaded head from {HEAD_BIN}")
        print(f"[INFO] classes={self.num_labels}, device={self.device}")
        print(f"[INFO] labels={self.labels}")
        print(f"[INFO] TOPK={TOPK}")

    @torch.inference_mode()
    def _encode(self, texts: List[str]):
        enc = self.tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.encoder.base_model(**enc)  # 取加入LoRA的 base_model
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return pooled

    @torch.inference_mode()
    def predict_one(self, text: str) -> Dict:
        pooled = self._encode([text])
        logits = self.head(pooled)  # [1, K]
        probs = torch.softmax(logits, dim=-1)[0]  # [K]

        # 主意图（唯一）
        pid = int(torch.argmax(probs).item())
        primary = (self.id2label[pid], float(probs[pid].item()))

        # TopK（主头）
        k = min(TOPK, self.num_labels)
        topk = torch.topk(probs, k=k)
        topk_pairs = [(self.id2label[idx.item()], float(topk.values[j].item())) for j, idx in enumerate(topk.indices)]

        return {
            "text": text,
            "primary": primary,
            "topk_primary": topk_pairs,
            "all_primary_probs": [(self.id2label[i], float(probs[i].item())) for i in range(self.num_labels)],
        }

# =======================
# 3) 交互主程序
# =======================
def main():
    clf = PrimaryInfer()
    print("\n================== 主意图推理 · 交互模式 ==================")
    print("输入中文句子并回车（exit / quit 退出）")
    print("==========================================================\n")

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] 已退出。")
            break

        if not text:
            continue
        low = text.lower()
        if low in {"exit", "quit", "q"}:
            print("[INFO] 已退出。")
            break

        out = clf.predict_one(text)
        pri_lbl, pri_p = out["primary"]
        p_top = " | ".join([f"{lbl}:{p:.3f}" for lbl, p in out["topk_primary"]])

        print(f"➡️ 主意图：{pri_lbl}（{pri_p:.3f}）")
        print(f"   Top{len(out['topk_primary'])}：{p_top}\n")

if __name__ == "__main__":
    main()
