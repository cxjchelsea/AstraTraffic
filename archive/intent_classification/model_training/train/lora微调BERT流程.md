## 🧠 一、你现在在做

> 🔹 使用 LoRA（Low-Rank Adaptation）
>  🔹 在预训练好的中文 BERT 模型上
>  🔹 进行「多标签意图识别」任务的下游微调。

换句话说，你的训练不是从零开始训练一个模型，
 而是让 BERT 学会你的**医疗意图分类任务**。

------

## ⚙️ 二、底座模型：BERT-base-chinese

### BERT 是什么？

BERT（Bidirectional Encoder Representations from Transformers）是一个双向 Transformer 编码器。
 它在大规模中文语料上预训练，已经“懂”了中文语义的基础结构，比如：

- 句法（谁是主语、宾语）
- 上下文依赖（“吃了药”意味着健康话题）
- 词义消歧（“感冒”是疾病，不是动词）

### 它预训练时做的任务：

1. **Masked Language Modeling (MLM)**：随机遮掉词，预测被遮的词。
2. **Next Sentence Prediction (NSP)**：预测两句话是否相邻。

所以预训练阶段它学的是「通用语言知识」。

------

## 🧩 三、Fine-tuning（微调）是什么？

预训练的 BERT 相当于一个“通用语义理解引擎”。
 当你要让它做特定任务（如医疗意图分类），
 你在它上面加一个「任务头（task head）」——
 比如一个 **分类层（Linear + Sigmoid）**。

然后：

1. 加载 BERT 的参数（被冻结或可微调）。
2. 加上任务特定的输出层（随机初始化）。
3. 用你的任务数据（主意图 + 次意图）训练几轮。

训练过程中，模型会更新最后几层的参数（或部分 LoRA 层），
 让 BERT 的语义空间**向你的任务分布靠拢**。

在你的 `train_intent_lora_multilabel.py` 里，有这样一段核心代码 👇：

```python
base = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_PATH,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
base.config.problem_type = "multi_label_classification"
```

这行做了两件事：

1. **加载底座模型：**
    这里的 `AutoModelForSequenceClassification` 会加载一个已经预训练好的 `BertForSequenceClassification`。
    它在原始 BERT 编码器顶部自动添加了一个任务层（classification head）。
2. **指定任务维度与任务类型：**
   - `num_labels=num_labels` 让最后一层的输出维度等于你的标签数（比如 10）。
   - `problem_type="multi_label_classification"` 告诉模型使用 **BCEWithLogitsLoss**，而不是 CrossEntropy。

也就是说：

> 你的 “任务头（task head）” 是在加载 `AutoModelForSequenceClassification` 时自动加上的。

```python
BertForSequenceClassification(
  (bert): BertModel(...)
  (dropout): Dropout(p=0.1)
  (classifier): Linear(in_features=768, out_features=10, bias=True)
)
```

其中 `(classifier)` 就是任务头。它接收 [CLS] 向量（即整个句子的语义表示）并输出 10 维 logits。

## 🧬 四、LoRA 是怎么做微调的？

传统 Fine-tuning 要更新整个 BERT（1亿参数+），成本高。

一个 BERT-base 模型 ≈ 1.1 亿参数，如果全部更新：

- 显存需求大（几 GB）
- 存储冗余（每个下游任务都得保存一份完整模型）
- 微调慢，容易灾难性遗忘

> 所以 LoRA 的目标是：**“只在关键权重上注入少量可训练参数”**，在不破坏预训练能力的前提下学到新任务。

假设 Transformer 里某个线性层权重矩阵是：

$W∈Rdout×dinW \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}W∈Rdout×din$

在原始微调里，我们要更新整个$ WWW$。

LoRA 的做法是：
 **保持原权重$ WWW$ 不动**，在它旁边加一个低秩矩阵调整项$ ΔW\Delta WΔW$：

W′=W+ΔW=W+BAW' = W + \Delta W = W + BAW′=W+ΔW=W+BA

其中：

- B∈Rdout×rB \in \mathbb{R}^{d_{\text{out}} \times r}B∈Rdout×r
- A∈Rr×dinA \in \mathbb{R}^{r \times d_{\text{in}}}A∈Rr×din
- r≪din,doutr \ll d_{\text{in}}, d_{\text{out}}r≪din,dout（例如 r = 4、8、16）

然后只训练 A 和 B，冻结 W。

> 🔹 **低秩** 意味着：你只在极小的参数子空间里学习任务特征。
>  例如，768×768 的全连接层要更新 ~60 万参数；而 r=8 的 LoRA 只要更新 768×8×2 = 1.2 万参数。

------

# 🧮 三、LoRA 的工作流程（前向 + 反向）

1. **前向传播**

   hout=(W+BA)hin=Whin+B(Ahin)h_{\text{out}} = (W + BA)h_{\text{in}} = Wh_{\text{in}} + B(Ah_{\text{in}})hout=(W+BA)hin=Whin+B(Ahin)

   - 第一项 `Wh_in` 是原模型的知识（冻结的预训练语义）；
   - 第二项 `B(Ah_in)` 是你新任务的调整（可训练）。

2. **反向传播**

   - 只更新 A、B；
   - WWW 保持不变；
   - 更新量极小，梯度传播快，占显存少。

3. **推理阶段**

   - 合并成一个等效矩阵 W′=W+α/r×BAW' = W + \alpha / r \times BAW′=W+α/r×BA；
   - 推理时速度几乎不变（可以直接 fuse 进原模型）。

------

# 🧩 四、LoRA 插入的位置（以 BERT 为例）

Transformer 的注意力子层结构：

```
Q = X * Wq
K = X * Wk
V = X * Wv
attention = softmax(QK^T)V
```

LoRA 通常插在这几个线性变换上：

- `query`（Wq）
- `key`（Wk）
- `value`（Wv）
- 有时还加在 `dense`（输出投影）

在你的代码中：

```
LORA_TARGETS = ["query", "key", "value", "dense"]
```

这意味着：

> LoRA 被注入到 BERT 的所有注意力线性层中（Query、Key、Value、Dense）。

每层都会有自己的 (A,B) 矩阵对。

------

# 🧠 五、LoRA 的主要可配置参数

在 Hugging Face 的 `peft.LoraConfig` 中，你能控制以下关键项 👇：

| 参数             | 含义                 | 示例值                            | 影响                                                    |
| ---------------- | -------------------- | --------------------------------- | ------------------------------------------------------- |
| `r`              | 低秩分解维度（rank） | 4 / 8 / 16                        | 越大 → 学习能力强、参数多；越小 → 模型更轻              |
| `lora_alpha`     | 缩放因子（scale）    | 16 / 32                           | 调整 LoRA 更新的影响力（类似 learning rate multiplier） |
| `lora_dropout`   | Dropout 概率         | 0.1                               | 在 LoRA 分支上施加 dropout，防止过拟合                  |
| `target_modules` | 注入位置             | `["query","key","value","dense"]` | 控制 LoRA 作用的线性层；不同任务可不同                  |
| `bias`           | 是否微调 bias 参数   | `"none"`, `"all"`, `"lora_only"`  | 一般 `"none"`；除非小模型才调 bias                      |
| `task_type`      | 任务类型             | `TaskType.SEQ_CLS`                | 决定兼容哪种模型（文本分类、生成、QA等）                |

------

# ⚖️ 六、在你的训练中，它们的实际取值是：

```
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGETS = ["query", "key", "value", "dense"]
```

意味着：

- 每个注意力线性层插入一个 rank=8 的 LoRA 分支；

- 更新参数只占 BERT 的约 1.3%；

- 有 0.1 的 dropout；

- LoRA 的影响力缩放为 α/r = 16/8 = 2；
   所以有效权重：

  W′=W+2×(B@A)W' = W + 2 \times (B @ A)W′=W+2×(B@A)

- 不更新 bias；

- 任务类型：Sequence Classification（文本分类）。

------

# 🧾 七、LoRA 在微调时“怎么与分类层配合”

> BERT 部分：冻结主干，LoRA 小幅调节注意力映射
>  分类头部分：随机初始化，正常全训练

训练的两个方向：

1. **分类层**：学习任务的判别边界（哪个句子属于哪个意图）
2. **LoRA 层**：微调 BERT 的注意力分布，让输入语义更贴近分类需求

最终效果：LoRA 改变了语义空间，使 `CLS` 向量在新任务上更可分。🧩 五、你的任务目标（多标签意图识别）

你让 BERT 学会：

> “一个句子中可能有多个医疗意图（主 + 次）。”

所以输出层是一个 **sigmoid 激活的多标签分类头**。
 每个标签（共10类）独立输出一个概率 ( p_i \in [0,1] )，
 损失函数是：

> **BCEWithLogitsLoss**（多标签交叉熵）

训练目标：
 让预测概率 ( p_i ) 尽可能贴近真实标签（0或1）。

------

## 🚀 六、训练过程里发生了什么？

每一步（step）大致是：

1. **Tokenizer** 把句子切成 token 序列，变成 `input_ids`。
2. BERT 的 embedding + Transformer 编码器计算句向量。
3. LoRA 插件更新 Wq/Wv 等的微小偏移。
4. 分类头输出 10 维 logits（每类一个）。
5. 对每个类计算 BCE loss。
6. 反向传播时，只更新 LoRA 层的 A/B 参数。
7. 记录指标：`loss`, `grad_norm`, `learning_rate`, `epoch`。

因为 LoRA 冻结了 99% 的参数，训练非常快，
 而且不会破坏原始 BERT 的语言能力。

------

## 📦 七、训练完后，你得到的文件

在 `models/bert-intent-lora-v1/` 下会看到：

```
adapter_config.json      # LoRA 配置 (r, alpha, dropout)
adapter_model.safetensors# LoRA 参数 (A/B矩阵)
tokenizer_config.json
special_tokens_map.json
vocab.txt
```

这意味着：

- 你不需要再保存 400MB 的 BERT 权重；
- 只要加载原 BERT + 这个 LoRA 适配器，就能完成推理。

好！咱们把“LoRA 在你这套训练代码里到底**怎么被实现**”说到代码级，按执行顺序拆开讲。你用的是 🤗 Transformers + PEFT（peft 库）。核心就是三步：

1. **配置 LoRA**（告诉它插到哪些层、秩 r、多大缩放等）
2. **把 LoRA“挂接”到 BERT**（返回一个带 LoRA 分支的模型）
3. **只训练 LoRA 分支＋分类头**（其它权重冻结），训练完只保存 LoRA 适配器

下面每一步我都给出对应的“你代码里做了什么”、“PEFT 在背后替你做了什么”、“必要时的底层公式/伪代码”。

------

# 1) 配置 LoRA（你写的）

```python
from peft import LoraConfig, TaskType, get_peft_model

lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                    # ← 低秩维度 rank
    lora_alpha=16,          # ← 缩放系数 α（实际缩放 α/r）
    lora_dropout=0.1,       # ← LoRA 分支上的 dropout
    bias="none",            # ← 不训练 bias
    target_modules=["query", "key", "value", "dense"],  # ← 要插LoRA的线性层名
)
```

- `target_modules` 用**子串匹配**方式对模块名筛选（例如 `...attention.self.query` 会命中 `"query"`）。
- `task_type=SEQ_CLS` 只是告诉 PEFT 你是分类任务，便于做默认适配。

------

# 2) 把 LoRA“挂”到 BERT（你写的）

```python
base = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_PATH,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
base.config.problem_type = "multi_label_classification"

model = get_peft_model(base, lora_cfg)
```

这里发生了两件关键事：

### 2.1 Transformers 侧

- `AutoModelForSequenceClassification(...)` = **BERT 编码器 + 分类头**（`classifier: Linear(768→num_labels)`）。
- `problem_type="multi_label_classification"` → 训练时会用 **BCEWithLogitsLoss**（多标签）。

### 2.2 PEFT 侧（`get_peft_model` 内做了什么）

- **扫描模型的子模块**，凡是名字里包含 `query`/`key`/`value`/`dense` 的 `nn.Linear`，替换为一个**带 LoRA 分支**的“包装线性层”（`LoraLinear`）。
- 对每个被命中的线性层 (W\in\mathbb{R}^{d_{out}\times d_{in}})：
  - 保留原权重 (W) **冻结**（`requires_grad=False`）
  - 新增两块可训练参数：
    - (A \in \mathbb{R}^{r\times d_{in}})（lora_A）
    - (B \in \mathbb{R}^{d_{out}\times r})（lora_B）
  - 前向计算改为：
     [
     y = XW^\top + \underbrace{\frac{\alpha}{r},X A^\top B^\top}_{\text{LoRA 分支}}
     ]
     也可以写成 (y = X(W + \tfrac{\alpha}{r}BA)^\top)。

**伪代码（PEFT 的 LoraLinear 做的事）：**

```python
def forward(x):
    base_out = x @ W.T                           # 冻结的原始线性层
    if training:
        x = dropout(x, p=lora_dropout)           # 只在 LoRA 分支上dropout
    lora_out = (x @ A.T) @ B.T                   # 低秩调整
    return base_out + (lora_alpha / r) * lora_out
```

你可以在训练脚本里打印一下，确认哪些层被替换、哪些参数在训练：

```python
for n, p in model.named_parameters():
    if p.requires_grad:
        print("TRAINABLE:", n, p.shape)
```

你会看到大量 `...query.lora_A.weight / lora_B.weight` 之类的条目，以及 `classifier.weight/bias`。

------

# 3) 只训练 LoRA 分支 + 分类头（你写的）

你没有手工冻结 BERT，因为 **PEFT 已经帮你冻结了**（被替换的原始 `nn.Linear` 权重 `requires_grad=False`），同时只让 LoRA 的 A/B **和** 任务的 `classifier` **是可训练的**。因此：

- **优化器**（Trainer 内部创建）只会抓取 `requires_grad=True` 的参数
- **反向传播**更新的只有：
  - 所有命中层的 `lora_A / lora_B`
  - 顶部 `classifier` 线性层（768→num_labels）

日志里这句就印证了：

```
Total params: 103,622,420 | Trainable: 1,347,082 (1.30%)
```

------

# 4) 损失函数与前向计算（你间接触发的）

你把：

```python
base.config.problem_type = "multi_label_classification"
```

设为多标签，`BertForSequenceClassification` 的 `forward` 会自动选用 **BCEWithLogitsLoss**。
 也就是对每个标签独立做 sigmoid + 二元交叉熵：

[
 \mathcal{L}=\frac{1}{K}\sum_{i=1}^K\Big(-y_i\log\sigma(z_i) - (1-y_i)\log(1-\sigma(z_i))\Big)
 ]

- 其中 (z_i) 是第 i 个标签的 logit（来自 `classifier`）
- (\sigma) 是 sigmoid
- (K) 是标签数（10）

> 这就是为什么我们把数据的标签做成 **multi-hot 向量**（主意图=1，所有次意图=1）。

------

# 5) 保存／加载（你写的）

训练完：

```python
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
```

PEFT 只会把 **LoRA 适配器** 存成：

```
adapter_config.json
adapter_model.safetensors
```

（体积很小）。**底座 BERT 不会再存一份**，因为你在推理时会单独加载它——

推理：

```python
base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_PATH, ...)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
# -> 把LoRA再挂上去，前向变成 W + α/r·BA
```

------

# 6) 你能“配置/改动”的 LoRA 参数

| 配置项           | 作用                       | 常用取值                               | 建议                                   |
| ---------------- | -------------------------- | -------------------------------------- | -------------------------------------- |
| `r`              | 低秩维度                   | 4 / 8 / 16                             | 任务更难/数据更大可加大；先用 8        |
| `lora_alpha`     | 缩放系数 α（实际缩放 α/r） | 16 / 32                                | 影响 LoRA 分支“力度”；16 搭配 r=8 常见 |
| `lora_dropout`   | LoRA 分支 dropout          | 0.0 ~ 0.1/0.3                          | 防过拟合；0.1 较稳                     |
| `target_modules` | 命中哪些线性层             | ["query","value"] / 加上 "key","dense" | 先 `q,v`，需要再加 `k,dense`           |
| `bias`           | 是否训练 bias              | "none"/"all"                           | 通常 "none"；小模型可试 "all"          |

**微调风格建议：**

- 想更轻：`r=4, targets=["query","value"]`
- 想更强：`r=16, targets=["query","key","value","dense"], alpha=32`
- 过拟合：`lora_dropout` 提高到 0.2–0.3；或减小 r；或加权重衰减

------

# 7) 两个小工具片段（可直接放进你的脚本调试）

**(1) 看看到底替换了哪些层：**

```python
print("===== LoRA modules =====")
for name, module in model.named_modules():
    if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
        print(name)
```

**(2) 统计可训练参数量：**

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / Total: {total:,} ({trainable/total*100:.2f}%)")
```

------

# 小结（把话说透）

- **你没有直接写“加任务头”的代码**，而是让 `AutoModelForSequenceClassification` 自动给 BERT **挂了一个 `classifier` 线性层**（任务头）。
- **你也没有手写“低秩分解”**，而是用 `get_peft_model` 让 PEFT 把 LoRA **注入到目标线性层**，把 (\Delta W = \tfrac{\alpha}{r}BA) 加到原权重上。
- 训练时，**只有 LoRA 的 A/B 和 `classifier` 在更新**；BERT 主体冻结。
- 因为你把 `num_train_epochs=4`，所以只训了 4 轮；LoRA 通常 3–5 轮就很稳。

如果你想，我也可以给你一个**“最小可跑的 LoRA 手写示例”**（不用 Trainer，直接 `optimizer.step()` 的 30 行玩具代码），把上面这些步骤在最小代码里跑一遍，帮助进一步理解 LoRA 的前向/反向到底发生了什么。需要的话告诉我就贴。