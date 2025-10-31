# -*- coding: utf-8 -*-
"""
rag_llm_cli.py
交互式：意图识别 → 多知识库检索(RAG) → LLM 生成（带引用）
- 命中为空/低置信：不生成，直接提示没有把握
- 输入 exit / quit / q 退出
"""
import argparse
from typing import List, Dict, Any, Optional

# ========= 1) 组件加载 =========
def load_components(device: Optional[str] = None, use_bm25: bool = True, use_reranker: bool = True):
    from process.intent_adapter import predict_intent
    from modules.retriever.rag_retriever import INTENT_TO_KB, KnowledgeSearcher, DenseEncoder, _build_reranker
    # 简单的 Searcher 缓存
    _cache = {}
    def get_searcher(kb_name: str):
        if kb_name in _cache:
            return _cache[kb_name]
        embedder = DenseEncoder(device=device)
        reranker = _build_reranker(device=device) if use_reranker else None
        searcher = KnowledgeSearcher(embedder, reranker, use_bm25=use_bm25, kb_name=kb_name)
        _cache[kb_name] = searcher
        return searcher
    return predict_intent, INTENT_TO_KB, get_searcher

# ========= 2) LLM 客户端（OpenAI 兼容 / Ollama / 自定义HTTP，自动适配）=========
import os, requests, time

def get_llm_mode():
    """
    从环境变量确定调用模式：
      LLM_MODE = openai | ollama | http
    默认优先 openai（若设置了 OPENAI_API_KEY），否则 ollama（本地11434），否则 http。
    """
    mode = os.getenv("LLM_MODE")
    if mode:
        return mode.lower()
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("LLM_API_BASE_URL", "http://localhost:11434"):
        return "ollama"
    return "http"


def get_llm_client():
    """
    返回一个统一的 callable: generate(prompt:str) -> str
    支持三种模式：
      - openai: 需 OPENAI_API_KEY（可选 OPENAI_BASE_URL），模型名用 RAG_LLM_MODEL
      - ollama: 默认 http://localhost:11434, 接口 /api/generate, 模型名用 RAG_LLM_MODEL（如 qwen2.5:14b）
      - http:   通用POST到 LLM_API_BASE_URL（需你自定义后端接收 {prompt, model}）
    """
    mode = get_llm_mode()
    model_name = os.getenv("RAG_LLM_MODEL", "qwen2.5:14b")

    if mode == "openai":
        from openai import OpenAI
        base_url = os.getenv("OPENAI_BASE_URL")  # 可为空
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

        def _gen(prompt: str, temperature=0.2, max_tokens=700):
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role":"user","content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()

        return _gen

    elif mode == "ollama":
        base_url = os.getenv("LLM_API_BASE_URL", "http://localhost:11434").rstrip("/")
        timeout = float(os.getenv("LLM_HTTP_TIMEOUT", "120"))

        def _gen(prompt: str, temperature=0.2, max_tokens=700):
            # Ollama 的 /api/generate 默认是流式；加 stream:false 得到完整文本
            payload = {
                "model": model_name,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                "stream": False
            }
            r = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            # Ollama 常见字段：response
            text = data.get("response") or data.get("text") or data.get("output") or ""
            return text.strip()

        return _gen

    else:  # 通用 HTTP：POST 到 LLM_API_BASE_URL
        base_url = os.getenv("LLM_API_BASE_URL")
        if not base_url:
            raise RuntimeError("LLM_MODE=http 时必须设置 LLM_API_BASE_URL")
        base_url = base_url.rstrip("/")
        timeout = float(os.getenv("LLM_HTTP_TIMEOUT", "120"))

        def _gen(prompt: str, temperature=0.2, max_tokens=700):
            payload = {"prompt": prompt, "model": model_name,
                       "temperature": temperature, "max_tokens": max_tokens}
            r = requests.post(base_url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            # 常见后端的几种字段名
            text = (data.get("text") or data.get("output") or data.get("data") or "").strip()
            return text

        return _gen

# ========= 3) Prompt 模板（更强防幻觉+结构化+引用）=========
def build_prompt(query: str, passages: List[Dict[str, Any]]) -> str:
    """
    结构化提示：
      - 只允许依据证据回答；不足则明确说无法确定
      - 先给要点，再给建议，最后列引用 [S1][S2]…
    """
    def norm(x: str) -> str:
        return (x or "").replace("\n", " ").strip()
    blocks = []
    for i, p in enumerate(passages, 1):
        blocks.append(f"[S{i}] {norm(p.get('text'))}  (source: {p.get('source','')})")
    context_block = "\n".join(blocks)

    return f"""你是一个严谨的中文医学助理。请仅依据“资料”作答，不得编造资料中没有的结论。
若资料不足以回答，请明确说明“根据现有资料无法给出确定答案”。

# 问题
{query}

# 资料（可能不完整）
{context_block}

# 作答要求（严格遵守）
1. 先用 3–5 句给出“核心结论”，语言简洁。
2. 若涉及用药或报告判读，必须提示“需结合个体情况并在医生指导下进行”。
3. 不要引用外部常识，**只**能使用资料中的信息。
4. 若无法确定，请直说无法确定，不要猜测。
5. 结尾给出“参考来源”，格式如：参考：[S1][S3]。
"""
# ========= 4) 生成答案（增加重试+空串兜底为抽取式）=========
def _extractive_fallback(query: str, hits: List[Dict[str, Any]], max_sents: int = 5) -> str:
    """
    简单抽取式兜底：从命中文段里抽取前几句，拼成回答，并追加引用。
    """
    import re
    def sent_split(t: str):
        t = (t or "").strip()
        parts = re.split(r"[。！？!?；;]\s*", t)
        return [p for p in parts if p]
    sents = []
    for h in hits:
        sents.extend(sent_split(h.get("text","")))
        if len(sents) >= max_sents:
            break
    sents = sents[:max_sents]
    refs = "参考：" + "".join([f"[S{i+1}]" for i in range(len(hits))]) if hits else ""
    if not sents:
        return "根据现有资料无法给出确定答案。"
    return "；".join(sents) + "。 " + refs


def generate_answer(query: str, hits: List[Dict[str, Any]]) -> str:
    """
    调用本地/远端 LLM 生成答案；若返回空或报错，回退到抽取式回答。
    """
    client = get_llm_client()
    prompt = build_prompt(query, hits)

    # 简单重试机制
    for attempt in range(2):
        try:
            text = client(prompt)  # 统一 callable
            if text and text.strip():
                return text.strip()
        except Exception as e:
            if attempt == 0:
                time.sleep(0.8)
                continue
            # 记录错误到控制台即可（也可以写日志）
            print(f"[LLM Error] {e}")

    # 兜底：抽取式回答
    return _extractive_fallback(query, hits)

# ========= 5) 主流程：意图 → 检索 → 判空 → 生成 =========
def answer_once(query: str,
                predict_intent,
                intent_to_kb: Dict[str, Optional[str]],
                get_searcher,
                conf_th: float = 0.4,
                top_k_final: int = 4) -> None:
    label, conf, topk = predict_intent(query)
    routed = label if conf >= conf_th else "闲聊其他"
    kb_name = intent_to_kb.get(routed, None)

    print(f"➡️ 主意图：{label}（{conf:.3f}）")
    if topk:
        tops = " | ".join([f"{l}:{p:.3f}" for l,p in topk[:5]])
        print(f"   TopK：{tops}")
    print("   路由到 KB：", kb_name or "不检索")

    if not kb_name:
        print("💬 当前意图不走 KB（通常交给 LLM 闲聊或其他业务），此处不演示。")
        return

    searcher = get_searcher(kb_name)
    hits = searcher.search(query, top_k_final=top_k_final)

    if not hits:
        print("🤔 我没有把握能从知识库里给出正确答案。建议换一种问法或补充更多信息。")
        return

    print("🔎 已检索到高置信片段，生成答案中…")
    ans = generate_answer(query, hits)
    print("\n====== 答案 ======")
    print(ans)
    print("\n====== 证据 ======")
    for i, h in enumerate(hits, 1):
        src = h.get("source", "")
        brief = h.get("text","").replace("\n"," ")
        if len(brief) > 160: brief = brief[:160] + "…"
        print(f"[S{i}] {brief}  ({src})")


# ========= 6) CLI =========
def main():
    parser = argparse.ArgumentParser("RAG+LLM 命令行（带意图识别与多库路由）")
    parser.add_argument("--device", type=str, default=None, help="cuda 或 cpu")
    parser.add_argument("--no_bm25", action="store_true", help="关闭 BM25")
    parser.add_argument("--no_reranker", action="store_true", help="关闭跨编码器重排")
    parser.add_argument("--conf_th", type=float, default=0.40, help="意图置信度阈值")
    parser.add_argument("--once", type=str, default=None, help="只答一次指定问题")
    args = parser.parse_args()

    predict_intent, intent_to_kb, get_searcher = load_components(
        device=args.device,
        use_bm25=(not args.no_bm25),
        use_reranker=(not args.no_reranker),
    )

    if args.once:
        answer_once(args.once, predict_intent, intent_to_kb, get_searcher, conf_th=args.conf_th)
        return

    print("\n================== RAG + LLM · 交互模式 ==================")
    print("输入问题回车；exit/quit/q 退出。")
    print("=========================================================\n")
    try:
        while True:
            q = input(">>> ").strip()
            if not q:
                continue
            if q.lower() in {"exit","quit","q"}:
                print("👋 再见！"); break
            answer_once(q, predict_intent, intent_to_kb, get_searcher, conf_th=args.conf_th)
            print()
    except (KeyboardInterrupt, EOFError):
        print("\n👋 再见！")

if __name__ == "__main__":
    # 需要 OPENAI_API_KEY；如用私有推理，设置 OPENAI_BASE_URL + 对应 key
    main()
