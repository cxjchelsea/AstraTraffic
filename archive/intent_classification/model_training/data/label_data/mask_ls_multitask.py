#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
mask_ls_tasks.py —— 产出与方案 A 完全匹配的 Label Studio 文件

特性：
- 输出主意图、次意图、多个同级意图
- 输出的 ls_config.xml 使用 $text / $weak_primary / $weak_secondary / $weak_same_level / $weak_confidence / $weak_reason
- 远端推理（OpenAI 兼容接口：Ollama / vLLM / 其它），并发+断点续跑
- 十类中文意图（主意图必选、次意图可空，同级意图可空）
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
OPENAI_BASE = "http://192.168.3.47:11434/v1"   # ← 你的远端服务地址（Ollama / vLLM）
OPENAI_API_KEY = "EMPTY"                       # Ollama 默认可任意占位；vLLM 如配置鉴权则填真实key
REMOTE_MODEL   = "qwen2.5:14b"                 # 与远端服务加载的模型名一致
MAX_TOKENS     = 32                           # 分类输出 JSON，64 足够；想更快可改 32
TEMPERATURE    = 0.0                           # 分类用贪心解码更稳
CONCURRENCY    = 8                             # 并发线程数，按远端负载调整
TIMEOUT_SEC    = 60                            # 单请求超时（可按需调整）
# ========================================================

# =============== 数据与输出 ==============================
DATASET_NAME   = "FreedomIntelligence/huatuo26M-testdatasets"
SPLIT: Optional[str] = None        # None=auto(train>test>validation)
QUESTION_FIELD = "questions"
MAX_SAMPLES: Optional[int] = None  # 先小样本验证可设 200；全量用 None
OUT_JSON   = "ls_tasks.json"       # 导入 Label Studio 的数组 JSON（带 annotations）
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
3) 同级意图是与主意图同等重要的其他意图，可以是多个，也可以为空数组。
4) 输出严格为 JSON，字段：primary(字符串,中文)、secondary(字符串数组,中文)、same_level(字符串数组,中文)、confidence(0~1)、reason(中文,不超过50字)。
5) 只输出 JSON，不要多余文本。
"""
USER_PROMPT_CN = "用户问题：{q}\n请按要求输出 JSON。"
JSON_REGEX = re.compile(r"\{.*\}", re.S)


# ----------------- 工具方法 -----------------
def load_input_dataset() -> Tuple[Dict[str, Any], str]:
    """加载数据集"""
    if os.path.isfile(DATASET_NAME) and DATASET_NAME.lower().endswith(".csv"):
        kwargs = {}
        ds_dict = load_dataset("csv", data_files=DATASET_NAME, **kwargs)
        split = pick_split(ds_dict)
        return ds_dict, split
    else:
        ds_dict = load_dataset(DATASET_NAME)
        split = SPLIT or pick_split(ds_dict)
        return ds_dict, split


def resolve_question_field(dset) -> str:
    """解析问题列名"""
    cols = list(dset.features.keys())
    if QUESTION_FIELD and QUESTION_FIELD in cols:
        return QUESTION_FIELD

    for k in QUESTION_FIELD:
        if k in cols:
            return k

    try:
        from datasets import Value
        for k, v in dset.features.items():
            if isinstance(v, Value) and v.dtype in ("string", "large_string"):
                return k
    except Exception:
        pass
    return cols[0] if cols else QUESTION_FIELD or "text"


def pick_split(ds_dict) -> str:
    keys = list(ds_dict.keys())
    for k in ("train", "test", "validation"):
        if k in keys:
            return k
    return keys[0]


def iter_questions(val: Any) -> Iterable[str]:
    """从 questions 字段抽问句：兼容多种格式"""
    keys = ("question", "instruction", "input", "query", "prompt")
    if val is None:
        return []
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    if isinstance(val, list):
        out = []
        for it in val:
            if isinstance(it, str) and it.strip():
                out.append(it.strip())
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


def safe_json_extract(text: str) -> Dict[str, Any]:
    """尽力从 text 中提取 JSON"""
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
    return {"primary": "闲聊其他", "secondary": [], "same_level": [], "confidence": 0.5, "reason": "解析失败兜底"}


def make_ls_result(primary: str, secondary: List[str], same_level: List[str]) -> List[Dict[str, Any]]:
    """生成 Label Studio 标注结果"""
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
    if same_level:
        res.append({
            "from_name": "same_level",
            "to_name": "text",
            "type": "choices",
            "value": {"choices": same_level},
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
                    txt = obj.get("data", {}).get("text")
                    if txt:
                        done.add(txt)
                except Exception:
                    continue
    return done

# 其它部分的代码...

def build_ls_config_xml() -> str:
    """Label Studio 配置（主意图单选必选 + 次意图多选可空 + 同级意图多选可空）——与英文键对齐"""
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

  <!-- 同级意图：可多选·可留空 -->
  <Header value="可选：请选择【同级意图】（可多选·可留空）" />
  <Choices name="same_level" toName="text" choice="multiple" showInLine="true">
    {choices_xml}
  </Choices>

  <!-- 预标注信息（仅展示，不参与 toName 引用） -->
  <Header value="—— 预标注（仅参考） ——" />
  <Text name="weak1" value="弱主意图：$weak_primary" />
  <Text name="weak2" value="弱次意图：$weak_secondary" />
  <Text name="weak3" value="弱同级意图：$weak_same_level" />
  <Text name="weak4" value="置信度：$weak_confidence" />
  <Text name="weak5" value="理由：$weak_reason" />
</View>"""

# 主流程...

# ----------------- 远程推理客户端 -----------------
client = OpenAI(
    base_url=OPENAI_BASE,
    api_key=OPENAI_API_KEY,
    timeout=TIMEOUT_SEC,
)

def classify_remote(q: str, retry: int = 3, backoff: float = 1.5) -> Tuple[str, List[str], List[str], float, str]:
    """调用远程推理接口进行分类"""
    system = {"role": "system", "content": LLM_SYSTEM_PROMPT_CN}
    user   = {"role": "user", "content": USER_PROMPT_CN.format(q=q)}
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
            same_level = data.get("same_level") or []

            if primary not in INTENTS_CN:
                primary = "闲聊其他"
            secondary = [s for s in secondary if s in INTENTS_CN]
            same_level = [s for s in same_level if s in INTENTS_CN]

            conf = float(data.get("confidence", 0.7))
            reason = str(data.get("reason", "")).strip()[:100]
            return (primary, secondary, same_level, max(0.0, min(1.0, conf)), reason)
        except (APIStatusError, RateLimitError, APIConnectionError, TimeoutError) as e:
            last_err = e
            time.sleep((k + 1) * backoff)
        except Exception as e:
            last_err = e
            time.sleep((k + 1) * backoff)
    return ("闲聊其他", [], [], 0.5, f"远端错误：{last_err}"[:100])


# ----------------- 主流程 -----------------
def main():
    print(f"📦 Loading dataset: {DATASET_NAME}")
    ds_dict, split = load_input_dataset()
    dset = ds_dict[split]
    print(f"✅ Using split: {split}, rows={len(dset)}; remote_model={REMOTE_MODEL}")

    q_field = resolve_question_field(dset)
    seen = set()
    questions = []
    for rec in dset:
        val = rec.get(q_field, None)
        for q in iter_questions(val):
            if q and q not in seen:
                seen.add(q)
                questions.append(q)
        if MAX_SAMPLES and len(questions) >= MAX_SAMPLES:
            break

    print(f"📝 Extracted questions: {len(questions)}")

    done = load_done_set(OUT_JSONL)
    if done:
        print(f"🔁 断点续跑：检测到已完成 {len(done)} 条，将跳过这些样本。")

    jsonl_f = open(OUT_JSONL, "a", encoding="utf-8")
    processed = 0
    start_time = time.time()

    to_process = [q for q in questions if q not in done]
    print(f"🚀 Start remote inference: {len(to_process)} to process, concurrency={CONCURRENCY}")
    buf_tasks: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        fut2q = {ex.submit(classify_remote, q): q for q in to_process}
        for fut in as_completed(fut2q):
            q = fut2q[fut]
            primary, secondary, same_level, conf, reason = fut.result()

            data = {
                "text": q,
                "weak_primary": primary,
                "weak_secondary": "、".join(secondary) if secondary else "",
                "weak_same_level": "、".join(same_level) if same_level else "",
                "weak_confidence": float(conf),
                "weak_reason": reason
            }
            task = {"data": data}
            task["annotations"] = [{
                "result": make_ls_result(primary, secondary, same_level),
                "was_cancelled": False,
                "ground_truth": False,
            }]

            jsonl_f.write(json.dumps(task, ensure_ascii=False) + "\n")
            jsonl_f.flush()

            buf_tasks.append(task)
            processed += 1

            if processed % 50 == 0:
                elapsed = time.time() - start_time
                speed = processed / max(1e-9, elapsed)
                print(f"🧩 Prelabelled {processed}/{len(to_process)} | {speed:.2f} samples/s | elapsed {elapsed/60:.1f} min")

    jsonl_f.close()

    tasks = []
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
    print(f"✅ 完成：处理 {processed}/{len(to_process)} 条 | 总耗时 {total_elapsed/60:.1f} 分钟 | 并发 {CONCURRENCY}")


if __name__ == "__main__":
    main()
