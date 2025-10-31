# -*- coding: utf-8 -*-
"""
多任务推理（主=单标签CE，次=多标签BCE）
- 加载：BERT底座 + LoRA适配器 + 两个头（heads.bin / heads_config.json）
- 交互输入一句中文，输出：
  * 主意图（唯一、置信度）
  * 次意图（≥阈值，可多个、置信度）
  * TopK 概览
"""

import os
import json
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# =======================
# 0) 固定配置（按需修改）
# =======================
BASE_MODEL_PATH = r"../models/bert-base-chinese"
ADAPTER_DIR     = r"../models/bert-intent-lora-v1/ds_multitask"  # 训练脚本输出目录
HEADS_BIN       = os.path.join(ADAPTER_DIR, "heads.bin")
HEADS_CFG       = os.path.join(ADAPTER_DIR, "heads_config.json")
LABEL_MAP_PATH  = r"../data/intent/label_map_infer.json"  # 可选；若找不到则从 heads_config 读取
MAX_LEN         = 256
TOPK            = 5
THRESHOLD       = 0.50                                      # 次意图阈值（≥阈值视为命中）

# 可选：不作为“次意图”输出（除非你想看全量）
BLOCK_SECONDARY = {"闲聊其他"}  # 可按需添加，如 {"闲聊其他", "系统操作"}

# =======================
# 1) 工具函数
# =======================
def load_label_map():
    # 优先用 label_map；否则用 heads_config 中的 id2label
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)
        if all(isinstance(v, int) for v in m.values()):
            label2id = {k: int(v) for k, v in m.items()}
            id2label = {v: k for k, v in label2id.items()}
        else:
            id2label = {int(k): v for k, v in m.items()}
            label2id = {v: k for k, v in id2label.items()}
        id2label = {i: id2label[i] for i in sorted(id2label)}
        return label2id, id2label
    # fallback: heads_config
    with open(HEADS_CFG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # heads_config.json 内保存的是 {"id2label": {0: "...", 1: "...", ...}}
    id2label = {int(k): v for k, v in cfg.get("id2label", {}).items()}
    id2label = {i: id2label[i] for i in sorted(id2label)}
    label2id = {v: k for k, v in id2label.items()}
    return label2id, id2label

# =======================
# 2) 构建推理模型（编码器+两个头）
# =======================
class MultiTaskHeads(nn.Module):
    """两个头：主（CE），次（BCE）"""
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.primary_head = nn.Linear(hidden_size, num_labels)   # 主：单标签
        self.secondary_head = nn.Linear(hidden_size, num_labels) # 次：多标签

    def forward(self, pooled):
        x = self.dropout(pooled)
        logits_p = self.primary_head(x)      # [N, K]
        logits_s = self.secondary_head(x)    # [N, K]
        return logits_p, logits_s

class MultiTaskInfer:
    def __init__(self):
        # 载入标签
        self.label2id, self.id2label = load_label_map()
        self.labels = [self.id2label[i] for i in range(len(self.id2label))]
        self.num_labels = len(self.labels)

        # 分词器
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)

        # 底座 + LoRA
        base = AutoModel.from_pretrained(BASE_MODEL_PATH)
        self.encoder = PeftModel.from_pretrained(base, ADAPTER_DIR)

        # 读取 heads 配置
        with open(HEADS_CFG, "r", encoding="utf-8") as f:
            heads_cfg = json.load(f)
        hidden = heads_cfg.get("hidden_size", base.config.hidden_size)
        if hidden != base.config.hidden_size:
            # 一般不会发生；这里仅做健壮性提示
            print(f"[WARN] heads_config hidden_size={hidden} 与 base hidden_size={base.config.hidden_size} 不一致，按 base 使用。")
            hidden = base.config.hidden_size

        # 构造 & 加载 heads
        self.heads = MultiTaskHeads(hidden, self.num_labels)
        state = torch.load(HEADS_BIN, map_location="cpu")
        self.heads.primary_head.load_state_dict(state["primary_head"])
        self.heads.secondary_head.load_state_dict(state["secondary_head"])

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device).eval()
        self.heads.to(self.device).eval()

        print(f"[INFO] Loaded encoder+LoRA from {ADAPTER_DIR}")
        print(f"[INFO] Loaded heads from {HEADS_BIN}")
        print(f"[INFO] classes={self.num_labels}, device={self.device}")
        print(f"[INFO] labels={self.labels}")
        print(f"[INFO] THRESHOLD(secondary)={THRESHOLD:.2f}, TOPK={TOPK}")

    @torch.inference_mode()
    def _encode(self, texts: List[str]):
        enc = self.tokenizer(
            texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.encoder.base_model(**enc)  # 注意：取的是底座（加入LoRA后的base_model）
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return pooled

    @torch.inference_mode()
    def predict_one(self, text: str) -> Dict:
        pooled = self._encode([text])                 # [1, hidden]
        logits_p, logits_s = self.heads(pooled)       # [1,K], [1,K]
        # 主（softmax）
        probs_p = torch.softmax(logits_p, dim=-1)[0]  # [K]
        # 次（sigmoid）
        probs_s = torch.sigmoid(logits_s)[0]          # [K]

        # 主意图（唯一）
        pid = int(torch.argmax(probs_p).item())
        primary = (self.id2label[pid], float(probs_p[pid].item()))

        # 次意图（≥阈值，且可过滤某些类）
        secondaries = []
        for i, p in enumerate(probs_s.tolist()):
            lbl = self.id2label[i]
            if lbl in BLOCK_SECONDARY:
                continue
            if p >= THRESHOLD and i != pid:          # 不把主意图再算进次
                secondaries.append((lbl, float(p)))
        secondaries.sort(key=lambda x: -x[1])

        # TopK 概览（来自“主头”的分布，便于观察主类信心）
        k = min(TOPK, self.num_labels)
        topk_p = torch.topk(probs_p, k=k)
        topk_pairs = [(self.id2label[idx.item()], float(topk_p.values[j].item())) for j, idx in enumerate(topk_p.indices)]

        # 也可输出次头的TopK（用于观察次意图概率排序）
        order_s = torch.argsort(probs_s, descending=True)
        topk_s_pairs = [(self.id2label[i.item()], float(probs_s[i].item())) for i in order_s[:k]]

        return {
            "text": text,
            "primary": primary,
            "secondary": secondaries,
            "topk_primary": topk_pairs,
            "topk_secondary": topk_s_pairs,
            "all_secondary_probs": [(self.id2label[i], float(probs_s[i].item())) for i in range(self.num_labels)]
        }

# =======================
# 3) 交互主程序
# =======================
def main():
    clf = MultiTaskInfer()
    print("\n================== 多任务推理 · 交互模式 ==================")
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
        sec_str = " | ".join([f"{lbl}({p:.3f})" for lbl, p in out["secondary"]]) or "（无）"
        p_top = " | ".join([f"{lbl}:{p:.3f}" for lbl, p in out["topk_primary"]])
        s_top = " | ".join([f"{lbl}:{p:.3f}" for lbl, p in out["topk_secondary"]])

        print(f"➡️ 主意图：{pri_lbl}（{pri_p:.3f}）")
        print(f"   次意图：{sec_str}")
        print(f"   Top{len(out['topk_primary'])}（主头）：{p_top}")
        print(f"   Top{len(out['topk_secondary'])}（次头）：{s_top}\n")

if __name__ == "__main__":
    main()
