# -*- coding: utf-8 -*-
"""
BERT + LoRA 主意图单任务微调（兼容老版本 transformers）
- 仅训练主意图；不包含次意图
- 共享编码器（BERT+LoRA）+ 单分类头（CrossEntropy）
- 自动适配旧版 TrainingArguments：不升级也能跑
"""

import os
import re
import json
import random
import inspect
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score

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
# 0) 路径与超参（按需修改）
# =========================
BASE_MODEL_PATH = r"../models/bert-base-chinese"
LABEL_MAP_PATH  = r"../data/intent/label_map_intent.json"
TRAIN_PATH      = r"../data/intent/ds_fixed_async_train.json"
DEV_PATH        = r"../data/intent/ds_fixed_async_dev.json"
TEST_PATH       = r"../data/intent/ds_fixed_async_test.json"
OUT_DIR         = r"../models/bert-intent-lora-v1/ds_single"

MAX_LEN         = 256
EPOCHS          = 4
LR              = 2e-4
BATCH_TRAIN     = 16
BATCH_EVAL      = 16
SEED            = 42

# LoRA
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.1
LORA_TARGETS    = ["query", "key", "value", "dense"]

# 标签别名（数据中的别名 → 标准名）
ALIAS = {
    "系统功能咨询": "系统操作",
    "系统设置": "系统操作",
    "功能咨询": "系统操作",
    # 如无需此条，删除即可
    "环境健康":   "日常生活",
}

# =========================
# 1) label_map 加载
# =========================
def load_label_map(fp: str) -> Tuple[Dict[str, int], Dict[int, str]]:
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

# =========================
# 2) 读取 & 统一数据结构（只保留主意图）
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

def normalize_items_primary_only(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    输出：{ "text": "...", "primary_intent": "..." }
    """
    out = []
    for x in items:
        text = x.get("text")
        if isinstance(x.get("data"), dict):
            text = text or x["data"].get("text")
        if not text:
            continue

        primary = None

        # Label Studio 强标注
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

        # weak_primary 兜底
        if not primary:
            data = x.get("data") or {}
            wp = data.get("weak_primary")
            if wp:
                primary = str(wp).strip()

        if primary:
            out.append({"text": str(text).strip(), "primary_intent": primary})
    return out

def load_split_primary(fp: str, label2id: Dict[str, int]) -> List[Dict[str, Any]]:
    raw = load_any_json(fp)
    data = normalize_items_primary_only(raw)
    unknown_p = sorted({
        d["primary_intent"] for d in data
        if normalize_label(d["primary_intent"], label2id) is None
    })
    print(f"[STAT] {os.path.basename(fp)} total={len(data)} | unknown_primary={unknown_p[:5]}")
    return data

# =========================
# 3) 构建监督数据（labels 为单标签ID）
# =========================
def to_supervised_primary(items: List[Dict[str, Any]], label2id: Dict[str, int]) -> Dataset:
    texts, y = [], []
    dropped = 0
    for x in items:
        p = x.get("primary_intent")
        p_norm = normalize_label(p, label2id) if p else None
        if p_norm is None:
            dropped += 1
            continue
        texts.append(x["text"])
        y.append(label2id[p_norm])
    if dropped > 0:
        print(f"[WARN] 丢弃无主意图或主意图未知样本 {dropped} 条。")
    return Dataset.from_dict({"text": texts, "labels": y})

# =========================
# 4) Collator
# =========================
class DataCollatorForPrimary(DataCollatorWithPadding):
    def __call__(self, features):
        labels = [f["labels"] for f in features]
        feats = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = super().__call__(feats)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

# =========================
# 5) 模型
# =========================
class BertForIntentPrimary(nn.Module):
    def __init__(self, encoder: PeftModel, hidden_size: int, num_labels: int):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.encoder.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        x = self.dropout(pooled)
        logits = self.classifier(x)
        loss = self.ce(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

# =========================
# 6) 主流程（自适应老版 TrainingArguments）
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
    train_raw = load_split_primary(TRAIN_PATH, label2id)
    dev_raw   = load_split_primary(DEV_PATH,   label2id)
    test_raw  = load_split_primary(TEST_PATH,  label2id)

    train_ds = to_supervised_primary(train_raw, label2id)
    dev_ds   = to_supervised_primary(dev_raw,   label2id) if len(dev_raw)  > 0 else None
    test_ds  = to_supervised_primary(test_raw,  label2id) if len(test_raw) > 0 else None
    print(f"[INFO] Datasets -> train={len(train_ds)}, dev={0 if dev_ds is None else len(dev_ds)}, test={0 if test_ds is None else len(test_ds)}")
    if len(train_ds) == 0:
        raise RuntimeError("[FATAL] 训练集为空。")

    # 分词
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
    def preprocess(batch):
        enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
        enc["labels"] = batch["labels"]
        return enc

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    dev_tok   = dev_ds.map(preprocess,   batched=True, remove_columns=dev_ds.column_names) if dev_ds is not None else None

    collator = DataCollatorForPrimary(tokenizer)

    # 编码器 + LoRA
    base_encoder = AutoModel.from_pretrained(BASE_MODEL_PATH)
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGETS,
    )
    peft_encoder = get_peft_model(base_encoder, lora_cfg)
    hidden = base_encoder.config.hidden_size

    # 模型
    model = BertForIntentPrimary(peft_encoder, hidden, K)

    # ===== 自适应 Trainer 参数 =====
    has_dev = dev_tok is not None and len(dev_tok) > 0
    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())

    # 先准备“新式”参数，再按支持情况筛选
    new_style_kwargs = dict(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        # 下两项在旧版可能不存在，稍后会过滤
        report_to="none",
        remove_unused_columns=False,
    )

    if has_dev:
        new_style_kwargs.update(
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="primary_macro_f1",
            greater_is_better=True,
        )
    else:
        new_style_kwargs.update(
            evaluation_strategy="no",
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=False,
        )

    # 过滤掉当前版本不支持的键
    filtered_kwargs = {k: v for k, v in new_style_kwargs.items() if k in supported}

    # 如果老版本不支持 evaluation_strategy/save_strategy，则回退
    if "evaluation_strategy" not in supported:
        # 老版本：使用 evaluate_during_training 控制评估
        if has_dev and "evaluate_during_training" in supported:
            filtered_kwargs["evaluate_during_training"] = True
        # 老版本不支持以下键，安全移除
        for k in ("load_best_model_at_end", "metric_for_best_model", "greater_is_better"):
            filtered_kwargs.pop(k, None)
        # eval_steps/save_steps 如果支持则可保留
        if "eval_steps" in supported and has_dev:
            filtered_kwargs["eval_steps"] = 200
        if "save_steps" in supported:
            filtered_kwargs["save_steps"] = 200 if has_dev else 500

    training_args = TrainingArguments(**filtered_kwargs)

    # 验证指标（主意图）
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        y_true = labels
        y_pred = np.argmax(logits, axis=1)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return {"primary_acc": acc, "primary_macro_f1": macro_f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok if has_dev else None,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics if has_dev else None,
    )

    # 训练
    trainer.train()

    # 保存 LoRA 适配器与主头
    model.encoder.save_pretrained(OUT_DIR)   # adapter_config.json + adapter_model.safetensors
    tokenizer.save_pretrained(OUT_DIR)

    head = {"primary_head": model.classifier.state_dict()}
    torch.save(head, os.path.join(OUT_DIR, "head.bin"))
    head_cfg = {
        "hidden_size": hidden,
        "num_labels": K,
        "id2label": {i: id2label[i] for i in range(K)}
    }
    with open(os.path.join(OUT_DIR, "head_config.json"), "w", encoding="utf-8") as f:
        json.dump(head_cfg, f, ensure_ascii=False, indent=2)

    print(f"[INFO] LoRA + primary head 已保存至：{OUT_DIR}")

    # ========== 训练后评估（dev/test）==========
    def evaluate_split(name: str, raw_ds: Dataset):
        if raw_ds is None or len(raw_ds) == 0:
            return
        print(f"\n[Eval:{name}] Evaluating primary ...")
        tok = raw_ds.map(preprocess, batched=True, remove_columns=raw_ds.column_names)
        dl = torch.utils.data.DataLoader(tok, batch_size=BATCH_EVAL, shuffle=False, collate_fn=collator)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device); model.eval()

        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in dl:
                for k in list(batch.keys()):
                    batch[k] = batch[k].to(device)
                out = model(**batch)
                logits = out["logits"].detach().cpu().numpy()
                y_true.append(batch["labels"].detach().cpu().numpy())
                y_pred.append(np.argmax(logits, axis=1))

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print({"primary_acc": round(acc, 4), "primary_macro_f1": round(f1m, 4)})

    evaluate_split("dev", dev_ds)
    evaluate_split("test", test_ds)

if __name__ == "__main__":
    main()
