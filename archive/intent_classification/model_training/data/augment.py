import os
import json
import uuid
import copy
import random
import asyncio
import aiohttp
import time
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

# =============== 配置区域（按需修改） ===============
FILES = ["data/intent/train.json", "data/intent/dev.json", "data/intent/test.json"]         # 源数据文件（也可只放一个 dataset.json）
PRIMARY_FROM_NAMES = {"primary", "主意图"}               # 作为“主意图”的 from_name
TARGET_PER_CLASS = 1000                                  # ⭐ 每个类别最终想要的固定条数
MAX_GEN_PER_SOURCE = 3                                   # 每条原样本最多生成多少条增强样本
RANDOM_SEED = 42
SPLIT_RATIOS = (0.7, 0.2, 0.1)
OUTPUT_PREFIX = "qwen_fixed_async"                       # 输出前缀：qwen_fixed_async_train.json 等

# —— 千问 / OpenAI 兼容接口 ——
API_BASE = "http://127.0.0.1:11434/v1"                    # 你的本地推理服务地址（保持 /v1）
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")      # 若不需要鉴权可随意
MODEL = "qwen2.5:14b"                           # 你的模型名
USE_CHAT_COMPLETIONS = True                              # True: /chat/completions；False: /completions

# 生成参数
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_TOKENS = 512

# 并发与重试
CONCURRENCY = 8             # 并发请求数
TIMEOUT_SEC = 120           # 单请求超时
MAX_RETRIES = 3             # 最大重试次数
BACKOFF_BASE = 1.5          # 退避系数
# ====================================================

random.seed(RANDOM_SEED)

# ----------- 数据读写与标签/文本抽取 -----------
def load_all(files: List[str]) -> List[dict]:
    data = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                part = json.load(f)
                if isinstance(part, list):
                    data.extend(part)
                else:
                    print(f"⚠️ {fp} 不是 JSON 数组，已跳过")
        except FileNotFoundError:
            print(f"⚠️ 文件未找到：{fp}")
    return data

def get_primary_labels(item: dict, primary_from_names: set) -> List[str]:
    labels = []
    for ann in item.get("annotations", []):
        for res in ann.get("result", []):
            if res.get("type") == "choices" and res.get("from_name") in primary_from_names:
                labels.extend(res.get("value", {}).get("choices", []))
    return labels

def get_text_field(item: dict) -> Tuple[str, str]:
    if isinstance(item.get(""), dict) and "text" in item["data"]:
        return "data.text", item["data"]["text"]
    if "text" in item:
        return "text", item["text"]
    if isinstance(item.get(""), dict):
        for k, v in item["data"].items():
            if isinstance(v, str):
                return f"data.{k}", v
    return "text", ""

def set_text_field(item: dict, path: str, new_text: str):
    if path.startswith("data.") and "data" in item and isinstance(item["data"], dict):
        key = path.split(".", 1)[1]
        item["data"][key] = new_text
    else:
        item["text"] = new_text

def clone_with_new_text(item: dict, new_text: str, new_id: int) -> dict:
    new_item = copy.deepcopy(item)
    new_item["id"] = new_id
    new_item["uid"] = str(uuid.uuid4())
    path, _ = get_text_field(new_item)
    set_text_field(new_item, path, new_text)
    for ann in new_item.get("annotations", []):
        ann["unique_id"] = str(uuid.uuid4())
    return new_item


# ---------------- Prompt 模板 ----------------
SYSTEM_PROMPT = (
    "你是数据增强助手。请改写用户给出的文本，保持主意图一致，不改变事实，不杜撰信息。\n"
    "要求：\n"
    "1) 不改变类别意图与关键信息；\n"
    "2) 适度同义替换、改写句式；\n"
    "3) 不添加与原文冲突的新事实；\n"
    "4) 语言自然流畅，保持原语言风格（中文/中英混排均可）。\n"
    "只输出增强后的文本，不要包含序号或解释。"
)

USER_PROMPT_TPL = (
    "请基于下述文本生成 {k} 条改写版本，保持主意图《{label}》不变：\n"
    "【原文】{text}\n"
    "输出格式：每行一个改写样本，不加序号。"
)

# ---------------- 异步 HTTP 调用 ----------------
def _headers():
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

async def _post_json(session: aiohttp.ClientSession, url: str, payload: dict, timeout: int) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(url, json=payload, headers=_headers(), timeout=timeout) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                return await resp.json()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = BACKOFF_BASE * (attempt + 1)
            print(f"⚠️ 请求失败（{e}），{delay:.1f}s 后重试...")
            await asyncio.sleep(delay)

async def call_qwen_generate_async(
    session: aiohttp.ClientSession,
    text: str,
    label: str,
    k: int
) -> List[str]:
    """一次调用尽量返回 k 条，若不足由上层决定是否再补。"""
    url = f"{API_BASE}/chat/completions" if USE_CHAT_COMPLETIONS else f"{API_BASE}/completions"
    prompt_user = USER_PROMPT_TPL.format(k=k, label=label, text=text)

    if USE_CHAT_COMPLETIONS:
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_user},
            ],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "n": 1
        }
    else:
        payload = {
            "model": MODEL,
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt_user}",
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "n": 1
        }

    data = await _post_json(session, url, payload, TIMEOUT_SEC)
    content = (
        data["choices"][0]["message"]["content"].strip()
        if USE_CHAT_COMPLETIONS else
        data["choices"][0]["text"].strip()
    )
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    uniq, seen = [], set()
    for ln in lines:
        if ln and ln != text and ln not in seen:
            uniq.append(ln)
            seen.add(ln)
    return uniq[:k]

async def call_qwen_generate_single_async(
    session: aiohttp.ClientSession,
    text: str,
    label: str
) -> str:
    """补齐时的单条调用。"""
    res = await call_qwen_generate_async(session, text, label, 1)
    return res[0] if res else ""

# -------------- 并发任务编排 --------------
class GenTask:
    """表示对某个源样本需要生成的条目数"""
    __slots__ = ("item", "label", "need")
    def __init__(self, item, label, need):
        self.item = item
        self.label = label
        self.need = need

async def process_task(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    task: GenTask,
    next_id_ref: dict
) -> List[dict]:
    """处理单个生成任务，返回生成的新样本列表"""
    out_items = []
    async with sem:
        path, src_text = get_text_field(task.item)
        if not src_text or task.need <= 0:
            return out_items

        # 先尝试一次性生成 need 条
        gens = await call_qwen_generate_async(session, src_text, task.label, task.need)

        # 若不足，则单条补齐
        while len(gens) < task.need:
            extra = await call_qwen_generate_single_async(session, src_text, task.label)
            if not extra:
                break
            gens.append(extra)

        # 转为样本对象
        for g in gens[:task.need]:
            nid = next_id_ref["id"]
            next_id_ref["id"] += 1
            out_items.append(clone_with_new_text(task.item, g, nid))
    return out_items

async def run_parallel_generation(tasks: List[GenTask], start_id: int) -> List[dict]:
    """并发执行所有生成任务"""
    connector = aiohttp.TCPConnector(limit=None)
    timeout = aiohttp.ClientTimeout(total=None)
    sem = asyncio.Semaphore(CONCURRENCY)
    next_id_ref = {"id": start_id}

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        coros = [process_task(sem, session, t, next_id_ref) for t in tasks]
        results = []
        done = 0
        for f in asyncio.as_completed(coros):
            try:
                res = await f
                results.extend(res)
            except Exception as e:
                print(f"❌ 子任务失败：{e}")
            finally:
                done += 1
                if done % 20 == 0:
                    print(f"…并发进度：{done}/{len(tasks)} 个任务完成")
        return results

# ----------- 7:2:1 拆分 -----------
def split_dataset(items: List[dict], ratios=(0.7, 0.2, 0.1)):
    random.shuffle(items)
    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * (ratios[0] + ratios[1]))
    return items[:n_train], items[n_train:n_val], items[n_val:]

# ------------------- 主流程 -------------------
def main():
    # 1) 读取数据
    data = load_all(FILES)
    if not data:
        print("❌ 未读取到任何样本。")
        return

    # 2) 按“第一主意图”分桶
    buckets = defaultdict(list)
    unlabeled = []
    for it in data:
        labs = get_primary_labels(it, PRIMARY_FROM_NAMES)
        if labs:
            buckets[labs[0]].append(it)
        else:
            unlabeled.append(it)

    # 打印原始分布
    print("\n📊 原始主意图分布：")
    for k, c in sorted(((k, len(v)) for k, v in buckets.items()), key=lambda x: -x[1]):
        print(f"  {k}: {c}")
    if unlabeled:
        print(f"  (无主意图样本): {len(unlabeled)} —— 不参与")

    # 3) 找最大 id
    max_id = 0
    for it in data:
        if isinstance(it.get("id"), int):
            max_id = max(max_id, it["id"])
    next_id = max_id + 1

    # 4) 规划：每类统一到 TARGET_PER_CLASS
    # 先确定每类需要“下采样/保留/生成”的策略
    result_all = []
    gen_tasks: List[GenTask] = []
    total_need = 0

    for label, items in buckets.items():
        cur = len(items)
        target = TARGET_PER_CLASS

        if cur > target:
            keep = random.sample(items, target)
            result_all.extend(keep)
            print(f"🔽 类别《{label}》下采样 {cur}→{target}，丢弃 {cur - target}")
        elif cur == target:
            result_all.extend(items)
            print(f"➖ 类别《{label}》保持 {cur}")
        else:
            need = target - cur
            print(f"🔧 类别《{label}》需补 {need} 条（Qwen 并行生成）")
            total_need += need

            # 将生成需求分配到源样本上，每条源样本不超过 MAX_GEN_PER_SOURCE
            i, produced = 0, 0
            while produced < need and items:
                src = items[i % len(items)]
                i += 1
                k = min(MAX_GEN_PER_SOURCE, need - produced)
                gen_tasks.append(GenTask(src, label, k))
                produced += k

            # 原始样本也加入
            result_all.extend(items)

    print(f"\n🧮 计划生成总量：{total_need} 条；并发度：{CONCURRENCY}")

    # 5) 并发生成
    start_t = time.time()
    new_items = asyncio.run(run_parallel_generation(gen_tasks, next_id))
    used = len(new_items)
    print(f"✅ 并发生成完成：新增 {used} 条，用时 {time.time() - start_t:.1f}s")

    # 6) 合并并做最终数量对齐（万一因失败/去重导致不足）
    result_all.extend(new_items)

    # 校验每类最终数量；若仍不足，只能提示（或再来一轮生成——此处只提示）
    final_buckets = defaultdict(list)
    for it in result_all:
        labs = get_primary_labels(it, PRIMARY_FROM_NAMES)
        if labs:
            final_buckets[labs[0]].append(it)

    warn_short = []
    for label, items in final_buckets.items():
        if len(items) < TARGET_PER_CLASS:
            warn_short.append((label, len(items)))
    if warn_short:
        print("\n⚠️ 下列类别未完全补足（可能因服务限速/文本过短/重复过滤）：")
        for lb, cnt in warn_short:
            print(f"  {lb}: {cnt}/{TARGET_PER_CLASS}")

    # 7) 7:2:1 拆分
    train, val, test = split_dataset(sum(final_buckets.values(), []), SPLIT_RATIOS)

    # 8) 输出文件
    with open(f"{OUTPUT_PREFIX}_train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(f"{OUTPUT_PREFIX}_val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)
    with open(f"{OUTPUT_PREFIX}_test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    # 9) 打印最终分布（全量）
    def count_primary(items):
        c = Counter()
        for it in items:
            labs = get_primary_labels(it, PRIMARY_FROM_NAMES)
            if labs:
                c.update([labs[0]])
        return c

    total_c = count_primary(sum(final_buckets.values(), []))
    print("\n📊 统一后（全量）主意图分布：")
    for k, v in sorted(total_c.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print("\n📦 已生成：")
    print(f"  {OUTPUT_PREFIX}_train.json  （{len(train)} 条）")
    print(f"  {OUTPUT_PREFIX}_dev.json    （{len(val)} 条）")
    print(f"  {OUTPUT_PREFIX}_test.json   （{len(test)} 条）")

if __name__ == "__main__":
    main()
