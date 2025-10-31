"""
编排核心：意图 → 路由 → 检索 → 阈值判空 → LLM 生成 → 引用/兜底
外部只需调用：rag_answer(query) -> AnswerPack
"""
from typing import Optional
from process.types import AnswerPack, IntentResult, IntentTop, Metrics
from modules.generator.prompt import build_prompt
from process.retriever_adapter import INTENT_TO_KB, retrieve
from process.generator_adapter import get_llm_client
from process.intent_adapter import predict_intent   # 你已有
from settings import CONF_TH, TOP_K_FINAL, USE_BM25, USE_RERANKER

def _extractive_fallback(hits, max_sents=5) -> str:
    import re
    def split(t): return [s for s in re.split(r"[。！？!?；;]\s*", (t or "")) if s]
    sents=[]
    for h in hits:
        sents += split(h.text)
        if len(sents) >= max_sents: break
    if not sents: return "根据现有资料无法给出确定答案。"
    body = "；".join(sents[:max_sents]) + "。"
    refs = "参考：" + "".join(f"[S{i+1}]" for i in range(len(hits))) if hits else ""
    return body + ("\n\n" + refs if refs else "")

def _postprocess_answer(text: str, hits) -> str:
    text = (text or "").strip()
    refs = "参考：" + "".join(f"[S{i+1}]" for i in range(len(hits))) if hits else ""
    if text and "参考：" not in text and refs:
        text += "\n\n" + refs
    return text
def rag_answer(query: str,
               *,
               device: Optional[str] = None,
               conf_th: float = CONF_TH,
               top_k_final: int = TOP_K_FINAL,
               use_bm25: bool = USE_BM25,
               use_reranker: bool = USE_RERANKER) -> AnswerPack:
    # 1) 意图
    lbl, conf, topk_raw = predict_intent(query)
    intent = IntentResult(label=lbl, score=conf, topk=[IntentTop(l, s) for l,s in (topk_raw or [])])

    routed = lbl if conf >= conf_th else "闲聊其他"
    kb = INTENT_TO_KB.get(routed, None)
    metrics = Metrics(used_kb=kb, intent_conf=conf, conf_th=conf_th, notes={})

    # 2) 非 KB 路由的意图：直接返回空 hits/answer
    if not kb:
        return AnswerPack(intent=intent, kb=None, hits=[], answer="", metrics=metrics)

    # 3) 检索
    hits = retrieve(query, kb, top_k_final=top_k_final, device=device, use_bm25=use_bm25, use_reranker=use_reranker)
    if not hits:
        return AnswerPack(intent=intent, kb=kb, hits=[], answer="", metrics=metrics)

    # 4) 生成（带兜底）
    prompt = build_prompt(query, hits)
    llm = get_llm_client()
    try:
        ans = llm(prompt).strip()
    except Exception as e:
        metrics.notes["llm_error"] = str(e)
        ans = ""
    if not ans:
        ans = _extractive_fallback(hits)
    ans = _postprocess_answer(ans, hits)

    return AnswerPack(intent=intent, kb=kb, hits=hits, answer=ans, metrics=metrics)
