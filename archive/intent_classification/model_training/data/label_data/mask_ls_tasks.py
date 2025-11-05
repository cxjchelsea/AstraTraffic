#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mask_ls_tasks.py —— 产出与方案 A 完全匹配的 Label Studio 文件

特性：
- 输出英文键 data：text / weak_primary / weak_secondary / weak_confidence / weak_reason
- 输出的 ls_config.xml 使用 $text / $weak_primary / $weak_secondary / $weak_confidence / $weak_reason
- 远端推理（OpenAI 兼容接口：Ollama / vLLM / 其它），并发+断点续跑
- 十类中文意图（主意图必选、次意图可空）
"""

from __future__ import annotations
import json
import os
import re
import time
from typing import Any, Iterable, List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from openai import OpenAI
from openai._exceptions import APIStatusError, RateLimitError, APIConnectionError

# --- 放在文件很前面，避免被后续导入覆盖 ---
import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

# =============== 推理服务配置（按需修改） =================
# 👉 如果是 Ollama：把 OPENAI_BASE 改成 http://<ip>:11434/v1 ，REMOTE_MODEL 改成 "qwen2.5:14b"
# OPENAI_BASE = "http://127.0.0.1:11434/v1"   # ← 你的远端服务地址（Ollama / vLLM）
OPENAI_BASE = "http://192.168.3.47:11434/v1"   # ← 你的远端服务地址（Ollama / vLLM）
OPENAI_API_KEY = "EMPTY"                       # Ollama 默认可任意占位；vLLM 如配置鉴权则填真实key
REMOTE_MODEL   = "qwen2.5:14b"                 # 与远端服务加载的模型名一致
MAX_TOKENS     = 32                           # 分类输出 JSON，64 足够；想更快可改 32
TEMPERATURE    = 0.0                           # 分类用贪心解码更稳
CONCURRENCY    = 8                             # 并发线程数，按远端负载调整
TIMEOUT_SEC    = 60                            # 单请求超时（可按需调整）
# ========================================================

# =============== 数据与输出 =============================
DATASET_NAME   = "FreedomIntelligence/huatuo26M-testdatasets"
SPLIT: Optional[str] = None        # None=auto(train>test>validation)
QUESTION_FIELD = "questions"
MAX_SAMPLES: Optional[int] = 10  # 先小样本验证可设 200；全量用 None
# # =============== CSV 兼容配置（新增） =====================
# DATASET_NAME   = "./label_data/cMedQA2/question.csv"
# # 如果 DATASET_NAME 是 *.csv，则按 CSV 读取
CSV_DELIMITER: Optional[str] = None   # 例："," 或 "\t"；None 表示自动
# SPLIT: Optional[str] = None        # None=auto(train>test>validation)
# # 当 QUESTION_FIELD 为空或不在列里时，按此候选列表自动探测
# QUESTION_FIELD = "content"
# MAX_SAMPLES: Optional[int] = None  # 先小样本验证可设 200；全量用 None
# =========================================================
OUT_JSON   = "test_ls_tasks.json"       # 导入 Label Studio 的数组 JSON（带 annotations）
OUT_JSONL  = "ls_tasks.jsonl"      # 断点续跑增量文件
OUT_CONFIG = "ls_config.xml"       # 与方案 A 匹配的配置
# ========================================================

# 十类中文意图
INTENTS_CN = [
    "健康咨询",
    "药品服务",
    "报告解读",
    "就医转诊",
    "紧急求助",
    "系统操作",
    "情感支持",
    "家庭管理",
    "环境健康",
    "闲聊其他",
]

# 让模型严格输出 JSON 的系统提示
LLM_SYSTEM_PROMPT_CN = f"""你是一个中文医疗意图分类助理。请基于用户一句话，在下列十个中文意图中判断：
{', '.join(INTENTS_CN)}

要求：
1) primary 必须且仅能从上面十类中选一个（中文标签，需完全匹配）。
2) secondary 为上面十类中的若干个（可为空数组），同样必须是中文标签。
3) 输出严格为 JSON，字段：primary(字符串,中文)、secondary(字符串数组,中文)、confidence(0~1)、reason(中文,不超过50字)。
4) 只输出 JSON，不要多余文本。
"""
USER_PROMPT_CN = "用户问题：{q}\n请按要求输出 JSON。"
JSON_REGEX = re.compile(r"\{.*\}", re.S)


# ----------------- 工具方法 -----------------
def load_input_dataset() -> Tuple[Dict[str, Any], str]:
    """
    返回 (DatasetDict, split_name)
    - 若 DATASET_NAME 是 CSV 路径：用 datasets.load_dataset('csv', ...)
    - 否则：按原逻辑 load_dataset(DATASET_NAME)
    """
    if os.path.isfile(DATASET_NAME) and DATASET_NAME.lower().endswith(".csv"):
        kwargs = {}
        if CSV_DELIMITER:
            kwargs["delimiter"] = CSV_DELIMITER
        ds_dict = load_dataset("csv", data_files=DATASET_NAME, **kwargs)
        # load_dataset('csv') 返回的 split 名一般是 'train'
        split = pick_split(ds_dict)
        return ds_dict, split
    else:
        ds_dict = load_dataset(DATASET_NAME)
        split = SPLIT or pick_split(ds_dict)
        return ds_dict, split


def resolve_question_field(dset) -> str:
    """
    确定用于抽问句的列名：
    1) 如果 QUESTION_FIELD 非空且存在，直接用
    2) 否则在常见列名里自动探测
    3) 否则选第一个 string 类型的列
    """
    cols = list(dset.features.keys())
    # 1) 直接命名优先
    if QUESTION_FIELD and QUESTION_FIELD in cols:
        return QUESTION_FIELD

    # 2) 常见候选
    for k in QUESTION_FIELD:
        if k in cols:
            return k

    # 3) 找第一个 string 列
    try:
        from datasets import Value
        for k, v in dset.features.items():
            if isinstance(v, Value) and v.dtype in ("string", "large_string"):
                return k
    except Exception:
        pass

    # 兜底：用第一个列名
    return cols[0] if cols else QUESTION_FIELD or "text"


def pick_split(ds_dict) -> str:
    keys = list(ds_dict.keys())
    for k in ("train", "test", "validation"):
        if k in keys:
            return k
    return keys[0]


def iter_questions(val: Any) -> Iterable[str]:
    """从 questions 字段抽问句：兼容 str / list[str] / list[dict] / dict"""
    keys = ("question", "instruction", "input", "query", "prompt")
    if val is None:
        return []
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    if isinstance(val, list):
        out: List[str] = []
        for it in val:
            if isinstance(it, str):
                s = it.strip()
                if s:
                    out.append(s)
            elif isinstance(it, dict):
                for k in keys:
                    v = it.get(k)
                    if isinstance(v, str) and v.strip():
                        out.append(v.strip())
                        break
        return out
    if isinstance(val, dict):
        for k in keys:
            v = val.get(k)
            if isinstance(v, str) and v.strip():
                return [v.strip()]
    return []


def build_ls_config_xml() -> str:
    """Label Studio 配置（主意图单选必选 + 次意图多选可空）——与英文键对齐"""
    choices_xml = "\n    ".join(f'<Choice value="{v}">{v}</Choice>' for v in INTENTS_CN)
    return f"""<View>
  <!-- 待标注文本（英文键 $text） -->
  <Text name="text" value="$text" />

  <!-- 主意图：必选·单选 -->
  <Header value="请选择【主意图】（必选·单选）" />
  <Choices name="primary" toName="text" choice="single" required="true" showInLine="true">
    {choices_xml}
  </Choices>

  <!-- 次意图：可多选·可留空 -->
  <Header value="可选：请选择【次意图】（可多选·可留空）" />
  <Choices name="secondary" toName="text" choice="multiple" showInLine="true">
    {choices_xml}
  </Choices>

  <!-- 预标注信息（仅展示，不参与 toName 引用） -->
  <Header value="—— 预标注（仅参考） ——" />
  <Text name="weak1" value="弱主意图：$weak_primary" />
  <Text name="weak2" value="弱次意图：$weak_secondary" />
  <Text name="weak3" value="置信度：$weak_confidence" />
  <Text name="weak4" value="理由：$weak_reason" />
</View>"""


def safe_json_extract(text: str) -> Dict[str, Any]:
    """尽力从 text 中提取 JSON；失败则返回兜底结构。"""
    m = JSON_REGEX.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    braces = re.findall(r"\{.*\}", text, re.S)
    if braces:
        try:
            return json.loads(braces[-1])
        except Exception:
            pass
    return {"primary": "闲聊其他", "secondary": [], "confidence": 0.5, "reason": "解析失败兜底"}


def make_ls_result(primary: str, secondary: List[str]) -> List[Dict[str, Any]]:
    res = []
    if primary:
        res.append({
            "from_name": "primary",
            "to_name": "text",
            "type": "choices",
            "value": {"choices": [primary]},
        })
    if secondary:
        res.append({
            "from_name": "secondary",
            "to_name": "text",
            "type": "choices",
            "value": {"choices": secondary},
        })
    return res


def load_done_set(jsonl_path: str) -> set:
    """从 JSONL 里加载已完成文本集合（断点续跑）"""
    done = set()
    if os.path.isfile(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    txt = obj.get("data", {}).get("text")   # 注意英文键
                    if txt:
                        done.add(txt)
                except Exception:
                    continue
    return done


# ----------------- 远程推理客户端 -----------------
client = OpenAI(
    base_url=OPENAI_BASE,
    api_key=OPENAI_API_KEY,
    timeout=TIMEOUT_SEC,
)

def classify_remote(q: str, retry: int = 3, backoff: float = 1.5) -> Tuple[str, List[str], float, str]:
    """
    调用远端 OpenAI 兼容接口（Chat Completions）做单条分类。
    返回：(primary, secondary[], confidence, reason)
    """
    system = {"role": "system", "content": LLM_SYSTEM_PROMPT_CN}
    user   = {"role": "user",   "content": USER_PROMPT_CN.format(q=q)}
    last_err = None
    for k in range(retry):
        try:
            rsp = client.chat.completions.create(
                model=REMOTE_MODEL,
                messages=[system, user],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = rsp.choices[0].message.content or ""
            data = safe_json_extract(text)

            primary = str(data.get("primary", "")).strip()
            secondary = data.get("secondary") or []
            if isinstance(secondary, str):
                secondary = [secondary]
            secondary = [s for s in (str(i).strip() for i in secondary) if s]

            # 约束到十类
            if primary not in INTENTS_CN:
                primary = "闲聊其他"
            secondary = [s for s in secondary if s in INTENTS_CN]

            try:
                conf = float(data.get("confidence", 0.7))
            except Exception:
                conf = 0.7
            reason = str(data.get("reason", "")).strip()[:100]
            return (primary, secondary, max(0.0, min(1.0, conf)), reason)
        except (APIStatusError, RateLimitError, APIConnectionError, TimeoutError) as e:
            last_err = e
            time.sleep((k + 1) * backoff)
        except Exception as e:
            last_err = e
            time.sleep((k + 1) * backoff)
    # 最终兜底
    return ("闲聊其他", [], 0.5, f"远端错误：{last_err}"[:100])


# ----------------- 主流程 -----------------
def main():
    print(f"📦 Loading dataset: {DATASET_NAME}")
    ds_dict, split = load_input_dataset()
    dset = ds_dict[split]
    print(f"✅ Using split: {split}, rows={len(dset)}; remote_model={REMOTE_MODEL}")

    # 解析问句列名（支持 CSV 自动探测）
    q_field = resolve_question_field(dset)
    if QUESTION_FIELD and QUESTION_FIELD != q_field:
        print(f"⚠️ 指定的 QUESTION_FIELD='{QUESTION_FIELD}' 不在数据列中，将使用自动识别列 '{q_field}'")
    else:
        print(f"🧭 Using question field: '{q_field}'")

    # 抽问句 + 去重
    seen = set()
    questions: List[str] = []
    for rec in dset:
        # 兼容 None / 空白
        val = rec.get(q_field, None)
        for q in iter_questions(val):
            if q and q not in seen:
                seen.add(q)
                questions.append(q)
        if MAX_SAMPLES and len(questions) >= MAX_SAMPLES:
            break

    if not questions:
        raise RuntimeError(f"未从列 '{q_field}' 抽取到问句。请检查 CSV 的列名或内容。")
    print(f"📝 Extracted questions: {len(questions)}")

    total = len(questions)
    print(f"📝 Extracted questions: {total}")

    # 断点续跑：加载已完成
    done = load_done_set(OUT_JSONL)
    if done:
        print(f"🔁 断点续跑：检测到已完成 {len(done)} 条，将跳过这些样本。")

    # 打开 JSONL 以便增量写
    jsonl_f = open(OUT_JSONL, "a", encoding="utf-8")

    processed = 0
    start_time = time.time()

    # 并发执行：跳过已完成的，分发到线程池
    to_process = [q for q in questions if q not in done]
    n = len(to_process)
    print(f"🚀 Start remote inference: {n} to process, concurrency={CONCURRENCY}")
    buf_tasks: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        fut2q = {ex.submit(classify_remote, q): q for q in to_process}
        for fut in as_completed(fut2q):
            q = fut2q[fut]
            primary, secondary, conf, reason = fut.result()

            data = {
                # 英文键 —— 与方案 A 的 XML 对齐
                "text": q,
                "weak_primary": primary,
                "weak_secondary": "、".join(secondary) if secondary else "",
                "weak_confidence": float(conf),
                "weak_reason": reason,
            }
            task: Dict[str, Any] = {"data": data}
            task["annotations"] = [{
                "result": make_ls_result(primary, secondary),
                "was_cancelled": False,
                "ground_truth": False,
            }]

            # 断点续跑：增量写入
            jsonl_f.write(json.dumps(task, ensure_ascii=False) + "\n")
            jsonl_f.flush()

            buf_tasks.append(task)
            processed += 1

            if processed % 50 == 0:
                elapsed = time.time() - start_time
                speed = processed / max(1e-9, elapsed)
                print(f"🧩 Prelabelled {processed}/{n} | {speed:.2f} samples/s | elapsed {elapsed/60:.1f} min")

    jsonl_f.close()

    # 汇总 JSONL → JSON（Label Studio 导入）
    tasks: List[Dict[str, Any]] = []
    with open(OUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                tasks.append(json.loads(line))
            except Exception:
                continue

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    print(f"💾 Wrote tasks (array JSON): {OUT_JSON}")

    with open(OUT_CONFIG, "w", encoding="utf-8") as f:
        f.write(build_ls_config_xml())
    print(f"⚙️  Wrote LS config: {OUT_CONFIG}")

    total_elapsed = time.time() - start_time
    print(f"✅ 完成：处理 {processed}/{n} 条（总样本 {total}）| 总耗时 {total_elapsed/60:.1f} 分钟 | 并发 {CONCURRENCY}")


if __name__ == "__main__":
    main()
