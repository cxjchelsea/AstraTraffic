# -*- coding: utf-8 -*-
"""
BERT + LoRA 多任务微调（主=单标签CE，次=多标签BCE）
- 数据：主意图必有；次意图可0~N（合规的Label Studio/JSON结构）
- 共享编码器（BERT+LoRA），两个头：
  * 主意图 (num_labels) -> CrossEntropy
  * 次意图 (num_labels) -> BCEWithLogits
- 训练集内评估主意图；训练结束后对 dev/test 评估主+次
- 保存：LoRA适配器 + heads.bin（两头参数）+ heads_config.json

环境要求：
  transformers>=4.30, peft>=0.6, datasets, scikit-learn, torch>=1.12
"""

import os
import re
import json
import random
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)

# =========================
# 0) 固定路径（按需修改）
# =========================
BASE_MODEL_PATH = r"../models/bert-base-chinese"  # 本地或HF名
LABEL_MAP_PATH  = r"../data/intent/label_map_intent.json"  # 10类映射（两种结构都兼容）
TRAIN_PATH      = r"../data/intent/ds_fixed_async_train.json"  # 训练集
DEV_PATH        = r"../data/intent/ds_fixed_async_dev.json"  # 验证集
TEST_PATH       = r"../data/intent/ds_fixed_async_test.json"  # 测试集（可为空）
OUT_DIR         = r"../models/bert-intent-lora-v1/ds_multitask"  # 输出目录

MAX_LEN         = 256
EPOCHS          = 4
LR              = 2e-4
BATCH_TRAIN     = 16
BATCH_EVAL      = 16
SEED            = 42

# LoRA 配置
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.1
LORA_TARGETS    = ["query", "key", "value", "dense"]

# 多任务损失加权：总损失 = loss_primary + LAMBDA * loss_secondary
LAMBDA          = 0.5

# 标签别名映射（数据中的别名 → 标准名）
ALIAS = {
    "系统功能咨询": "系统操作",
    "系统设置": "系统操作",
    "功能咨询": "系统操作",
    "环境健康":   "日常生活",
}

# =========================
# 1) 工具：加载 label_map
# =========================
def load_label_map(fp: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(fp, "r", encoding="utf-8") as f:
        m = json.load(f)
    if all(isinstance(v, int) for v in m.values()):
        label2id = {k: int(v) for k, v in m.items()}
        id2label = {v: k for k, v in label2id.items()}
    else:
        id2label = {int(k): v for k, v in m.items()}
        id2label = {i: id2label[i] for i in sorted(id2label.keys())}
        label2id = {v: k for k, v in id2label.items()}
    return label2id, id2label

def normalize_label(lbl: str, label2id: Dict[str, int]) -> str:
    if lbl in label2id:
        return lbl
    if lbl in ALIAS and ALIAS[lbl] in label2id:
        return ALIAS[lbl]
    return None

# =========================
# 2) 读取 & 统一数据结构
# =========================
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
    输出统一结构：
      {
        "text": "...",
        "primary_intent": "...",          # 必有（强标注或 weak_primary）
        "secondary_intents": ["...", ...] # 可空，可多个
      }
    """
    out = []
    for x in items:
        # 文本
        text = x.get("text")
        if isinstance(x.get("data"), dict):
            text = text or x["data"].get("text")
        if not text:
            continue

        primary = None
        secondaries = []

        # 强标注（Label Studio）
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

def load_split(fp: str, label2id: Dict[str, int]) -> List[Dict[str, Any]]:
    raw = load_any_json(fp)
    data = normalize_items(raw)

    unknown_p = sorted({
        d["primary_intent"] for d in data
        if "primary_intent" in d and normalize_label(d["primary_intent"], label2id) is None
    })
    unknown_s = sorted({
        s for d in data for s in d.get("secondary_intents", []) or []
        if normalize_label(s, label2id) is None
    })
    print(f"[STAT] {os.path.basename(fp)} total={len(data)} | unknown_primary={unknown_p[:5]} | unknown_secondary={unknown_s[:5]}")
    return data

# =========================
# 3) 监督样本：主=单标签ID，次=multi-hot
# =========================
def to_multitask_supervised(items: List[Dict[str, Any]], label2id: Dict[str, int]) -> Dataset:
    texts, y_primary, y_secondary = [], [], []
    K = len(label2id)
    dropped = 0

    for x in items:
        if "text" not in x:
            continue

        # 主（必须）
        p = x.get("primary_intent")
        p_norm = normalize_label(p, label2id) if p else None
        if p_norm is None:
            dropped += 1
            continue
        pid = label2id[p_norm]

        # 次（可0~N）
        multi = [0] * K
        for s in x.get("secondary_intents", []) or []:
            s_norm = normalize_label(s, label2id)
            if s_norm:
                multi[label2id[s_norm]] = 1

        texts.append(x["text"])
        y_primary.append(pid)     # int
        y_secondary.append(multi) # list[int]

    if dropped > 0:
        print(f"[WARN] 丢弃无主意图或主意图未知样本 {dropped} 条。")
    return Dataset.from_dict({"text": texts, "labels_primary": y_primary, "labels_secondary": y_secondary})

# =========================
# 4) 数据整理器：保证标签 dtype 正确
# =========================
class DataCollatorForMultiTask(DataCollatorWithPadding):
    def __call__(self, features):
        lp = [f["labels_primary"] for f in features]
        ls = [f["labels_secondary"] for f in features]
        feats = [{k: v for k, v in f.items() if k not in ("labels_primary", "labels_secondary")} for f in features]
        batch = super().__call__(feats)
        batch["labels_primary"] = torch.tensor(lp, dtype=torch.long)    # CE 需要 long
        batch["labels_secondary"] = torch.tensor(ls, dtype=torch.float) # BCE 需要 float
        return batch

# =========================
# 5) 模型：共享编码器 + 两个头
# =========================
class BertForIntentMultiTask(nn.Module):
    def __init__(self, encoder: PeftModel, hidden_size: int, num_labels: int):
        super().__init__()
        self.encoder = encoder               # 已套LoRA的 BERT 编码器 (AutoModel)
        self.num_labels = num_labels

        self.dropout = nn.Dropout(0.1)
        self.primary_head = nn.Linear(hidden_size, num_labels)   # 主：单标签CE
        self.secondary_head = nn.Linear(hidden_size, num_labels) # 次：多标签BCE

        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels_primary=None,        # int64 [N]
        labels_secondary=None,      # float [N,K]
    ):
        outputs = self.encoder.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # 取 pooled：优先pooler_output，否则CLS
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]

        x = self.dropout(pooled)
        logits_p = self.primary_head(x)     # [N,K]
        logits_s = self.secondary_head(x)   # [N,K]

        loss = None
        if labels_primary is not None and labels_secondary is not None:
            loss_p = self.ce(logits_p, labels_primary)
            loss_s = self.bce(logits_s, labels_secondary)
            loss = loss_p + LAMBDA * loss_s

        # 为了 Trainer 兼容：提供 logits（用主头logits），并额外暴露两个头
        return {
            "loss": loss,
            "logits": logits_p,               # 供Trainer默认拿来做eval_pred
            "logits_primary": logits_p,
            "logits_secondary": logits_s,
        }

# =========================
# 6) 主流程
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # 标签
    label2id, id2label = load_label_map(LABEL_MAP_PATH)
    K = len(id2label)
    print(f"[INFO] Loaded {K} labels: {[id2label[i] for i in range(K)]}")
    if K == 0:
        raise ValueError("[ERR] label_map 为空。")

    # 数据
    train_raw = load_split(TRAIN_PATH, label2id)
    dev_raw   = load_split(DEV_PATH,   label2id)
    test_raw  = load_split(TEST_PATH,  label2id)

    train_ds = to_multitask_supervised(train_raw, label2id)
    dev_ds   = to_multitask_supervised(dev_raw,   label2id) if len(dev_raw)  > 0 else None
    test_ds  = to_multitask_supervised(test_raw,  label2id) if len(test_raw) > 0 else None
    print(f"[INFO] Datasets -> train={len(train_ds)}, dev={0 if dev_ds is None else len(dev_ds)}, test={0 if test_ds is None else len(test_ds)}")
    if len(train_ds) == 0:
        raise RuntimeError("[FATAL] 训练集为空。")

    # 分词
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
    def preprocess(batch):
        enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
        enc["labels_primary"] = batch["labels_primary"]
        enc["labels_secondary"] = batch["labels_secondary"]
        return enc

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    dev_tok   = dev_ds.map(preprocess,   batched=True, remove_columns=dev_ds.column_names) if dev_ds is not None else None

    collator = DataCollatorForMultiTask(tokenizer)

    # 编码器 + LoRA
    base_encoder = AutoModel.from_pretrained(BASE_MODEL_PATH)
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,   # 编码器场景（非直接seq_cls）
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGETS,
    )
    peft_encoder = get_peft_model(base_encoder, lora_cfg)

    # 取 hidden_size
    hidden = base_encoder.config.hidden_size

    # 多任务模型（共享 peft_encoder）
    model = BertForIntentMultiTask(peft_encoder, hidden, K)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    # 只在训练过程里度量“主意图”指标（次意图评估放在训练后）
    def compute_metrics(eval_pred):
        logits_p, labels = eval_pred
        # 注意：Trainer默认把 `model output["logits"]` 作为 predictions（这里就是主头）
        y_true = labels  # 这是 labels_primary（因为 collator把两个labels都传了，但默认只取第一个）
        y_pred = np.argmax(logits_p, axis=1)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return {"primary_acc": acc, "primary_macro_f1": macro_f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics if dev_tok is not None else None,
    )

    # 训练
    trainer.train()

    # 保存 LoRA 适配器（encoder）与 heads
    # peft_encoder 在 model.encoder 内部
    model.encoder.save_pretrained(OUT_DIR)   # 这会生成 adapter_config.json + adapter_model.safetensors
    tokenizer.save_pretrained(OUT_DIR)
    # 保存两个头
    heads = {
        "primary_head": model.primary_head.state_dict(),
        "secondary_head": model.secondary_head.state_dict(),
    }
    torch.save(heads, os.path.join(OUT_DIR, "heads.bin"))
    # 保存 heads 配置（便于推理脚本恢复）
    heads_cfg = {"hidden_size": hidden, "num_labels": K, "id2label": {i: id for i, id in enumerate([id2label[i] for i in range(K)])}}
    with open(os.path.join(OUT_DIR, "heads_config.json"), "w", encoding="utf-8") as f:
        json.dump(heads_cfg, f, ensure_ascii=False, indent=2)

    print(f"[INFO] LoRA + heads 已保存至：{OUT_DIR}")

    # =====================
    # 训练后评估（dev/test）：主+次
    # =====================
    def evaluate_split(name: str, raw_ds: Dataset):
        if raw_ds is None or len(raw_ds) == 0:
            return
        print(f"\n[Eval:{name}] Evaluating primary & secondary ...")

        tok = raw_ds.map(preprocess, batched=True, remove_columns=raw_ds.column_names)
        dl = torch.utils.data.DataLoader(tok, batch_size=BATCH_EVAL, shuffle=False, collate_fn=collator)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device); model.eval()

        y_p_true, y_p_pred = [], []
        y_s_true, y_s_pred = [], []

        with torch.no_grad():
            for batch in dl:
                for k in list(batch.keys()):
                    batch[k] = batch[k].to(device)
                out = model(**batch)
                lp = out["logits_primary"].detach().cpu().numpy()
                ls = out["logits_secondary"].detach().cpu().numpy()
                # 主：单标签
                y_p_true.append(batch["labels_primary"].detach().cpu().numpy())
                y_p_pred.append(np.argmax(lp, axis=1))
                # 次：多标签（阈值=0.5）
                probs = 1/(1+np.exp(-ls))
                y_s_true.append(batch["labels_secondary"].detach().cpu().numpy())
                y_s_pred.append((probs >= 0.5).astype(int))

        y_p_true = np.concatenate(y_p_true)
        y_p_pred = np.concatenate(y_p_pred)
        y_s_true = np.concatenate(y_s_true, axis=0)
        y_s_pred = np.concatenate(y_s_pred, axis=0)

        # 主意图指标
        p_acc = accuracy_score(y_p_true, y_p_pred)
        p_f1m = f1_score(y_p_true, y_p_pred, average="macro", zero_division=0)
        print({"primary_acc": round(p_acc, 4), "primary_macro_f1": round(p_f1m, 4)})

        # 次意图指标
        s_micro = f1_score(y_s_true, y_s_pred, average="micro", zero_division=0)
        s_macro = f1_score(y_s_true, y_s_pred, average="macro", zero_division=0)
        print({"secondary_micro_f1": round(s_micro, 4), "secondary_macro_f1": round(s_macro, 4)})

        # 次意图每类
        p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(y_s_true, y_s_pred, average=None, zero_division=0)
        labels = [id2label[i] for i in range(K)]
        print("[Secondary per-class]")
        for i, name_i in enumerate(labels):
            print(f"{name_i: <8}  P={p_c[i]:.3f}  R={r_c[i]:.3f}  F1={f1_c[i]:.3f}  supp={int(sup_c[i])}")

    # 评估 dev / test
    evaluate_split("dev", dev_ds)
    evaluate_split("test", test_ds)

if __name__ == "__main__":
    main()
