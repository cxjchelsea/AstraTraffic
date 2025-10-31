# rag_core.py
# -*- coding: utf-8 -*-
"""
编排核心：意图 → 路由 → 检索 → 质量判定 → LLM 生成 → 引用/兜底
外部只需调用：rag_answer(query) -> AnswerPack
"""
from typing import Optional, List, Tuple
import math

from process.types import AnswerPack, IntentResult, IntentTop, Metrics
from modules.generator.prompt import build_prompt
from process.retriever_adapter import INTENT_TO_KB, retrieve
from process.generator_adapter import get_llm_client
from process.intent_adapter import predict_intent
from settings import CONF_TH, TOP_K_FINAL, USE_BM25, USE_RERANKER


# -------- 交通关键词兜底路由（当意图置信度低或未匹配时）--------
_FALLBACK_KEYWORDS = [
    (("限行", "处罚", "罚款", "记分", "专用道", "电子警察"), "law"),
    (("信号", "配时", "相位", "绿信比", "潮汐车道", "可变车道", "诱导"), "handbook"),
    (("公交", "地铁", "轨道", "换乘", "票价", "首班", "末班", "到站"), "transit"),
    (("停车", "泊位", "车场", "路侧"), "parking"),
    (("充电", "快充", "直流", "交流", "电桩"), "ev"),
    (("车路协同", "RSU", "OBU", "C-V2X", "V2X"), "iov"),
]

def _keyword_route(query: str) -> Optional[str]:
    q = (query or "").lower()
    for kws, kb in _FALLBACK_KEYWORDS:
        for k in kws:
            if k.lower() in q:
                return kb
    return None


# -------- 抽取式兜底（无 LLM 或 LLM 失败时）--------
def _extractive_fallback(hits, max_sents=5) -> str:
    import re
    def split(t): return [s for s in re.split(r"[。！？!?；;]\s*", (t or "")) if s]
    sents = []
    for h in hits:
        sents += split(h.text)
        if len(sents) >= max_sents:
            break
    if not sents:
        return "根据现有资料无法给出确定答案。"
    body = "；".join(sents[:max_sents]) + "。"
    refs = "参考：" + "".join(f"[S{i+1}]" for i in range(len(hits))) if hits else ""
    return body + ("\n\n" + refs if refs else "")


def _postprocess_answer(text: str, hits) -> str:
    text = (text or "").strip()
    refs = "参考：" + "".join(f"[S{i+1}]" for i in range(len(hits))) if hits else ""
    if text and "参考：" not in text and refs:
        text += "\n\n" + refs
    return text


# -------- 命中质量判定：低分/分散 → 判空 --------
def _hits_quality_ok(hits, min_top_score: float = 0.15, min_margin: float = 0.03) -> bool:
    """
    - min_top_score：Top1 过低 → 可能不相关
    - min_margin   ：Top1 与 Top2 差距过小 → 模糊不清
    注意：score 未必是同一量纲，这里仅做启发式门槛。
    """
    if not hits:
        return False
    scores = [float(getattr(h, "score", 0.0)) for h in hits]
    top = scores[0]
    if top < min_top_score:
        return False
    if len(scores) >= 2 and (top - scores[1]) < min_margin:
        # 分散度高，可能主题不聚焦
        pass  # 你也可以直接 return False，这里放宽一点
    return True


# -------- 非 KB 意图的友好答复 --------
def _no_kb_response(intent_label: str) -> str:
    if intent_label in ("闲聊其他", "系统操作"):
        return "这是智慧交通助手。如果你想查询限行规则、信号配时、公交换乘、停车计费或充电规范，请描述更具体的道路/线路/时段/场景。"
    # 未配置路由的交通意图
    return "该问题暂未接入知识库。请补充更具体的信息（如道路/路段、时间段、线路名），或改问法规/配时/公交/停车/充电/车路协同等主题。"


def rag_answer(
    query: str,
    *,
    device: Optional[str] = None,
    conf_th: float = CONF_TH,
    top_k_final: int = TOP_K_FINAL,
    use_bm25: bool = USE_BM25,
    use_reranker: bool = USE_RERANKER
) -> AnswerPack:
    # 1) 意图识别
    lbl, conf, topk_raw = predict_intent(query)
    intent = IntentResult(label=lbl, score=conf, topk=[IntentTop(l, s) for l, s in (topk_raw or [])])

    # 2) 路由：高置信 → 按 INTENT_TO_KB；低置信 → 关键词兜底；仍无 → 视为无 KB
    if conf >= conf_th:
        routed = lbl
        kb = INTENT_TO_KB.get(routed, None)
    else:
        kb = _keyword_route(query) or INTENT_TO_KB.get("闲聊其他", None)

    metrics = Metrics(used_kb=kb, intent_conf=conf, conf_th=conf_th, notes={})

    # 3) 非 KB 路由：返回友好提示（不空答）
    if not kb:
        return AnswerPack(
            intent=intent,
            kb=None,
            hits=[],
            answer=_no_kb_response(lbl),
            metrics=metrics
        )

    # 4) 检索
    hits = retrieve(
        query, kb,
        top_k_final=top_k_final,
        device=device,
        use_bm25=use_bm25,
        use_reranker=use_reranker
    )

    # 5) 命中质量判定（可调阈值）
    if not _hits_quality_ok(hits):
        return AnswerPack(
            intent=intent,
            kb=kb,
            hits=[],
            answer="知识库里没有找到足够可靠的资料。请补充更具体的信息（道路/路段、时间段、线路名），我再查一次；或改问法规/配时/公交/停车/充电/车路协同等主题。",
            metrics=metrics
        )

    # 6) 生成（LLM）+ 兜底（抽取）
    prompt = build_prompt(query, hits)
    llm = get_llm_client()
    try:
        ans = (llm(prompt) or "").strip()
    except Exception as e:
        metrics.notes["llm_error"] = str(e)
        ans = ""

    if not ans:
        ans = _extractive_fallback(hits)

    ans = _postprocess_answer(ans, hits)

    return AnswerPack(
        intent=intent,
        kb=kb,
        hits=hits,
        answer=ans,
        metrics=metrics
    )
