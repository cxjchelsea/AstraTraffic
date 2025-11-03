# -*- coding: utf-8 -*-
"""
基于 LangChain 的 RAG Chain 编排
使用 LangChain 的 LCEL (LangChain Expression Language) 构建 RAG 流程
"""
from typing import Optional, Dict, Any
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from lc.retriever_adapter import IntentAwareRetriever, CustomRetriever
from lc.llm_adapter import get_langchain_llm
from lc.prompt_adapter import create_rag_prompt, format_context
from lc.intent_adapter import predict_intent
from rag_types import AnswerPack, IntentResult, IntentTop, Metrics, Hit
from settings import USE_BM25, USE_RERANKER, TOP_K_FINAL, CONF_TH


def _hits_quality_ok(documents: list[Document], min_top_score: float = 0.15, min_margin: float = 0.03) -> bool:
    """判断检索结果质量是否合格"""
    if not documents:
        return False
    
    scores = [float(doc.metadata.get("score", 0.0)) for doc in documents]
    top = scores[0]
    
    if top < min_top_score:
        return False
    
    if len(scores) >= 2 and (top - scores[1]) < min_margin:
        pass  # 可以放宽一点
    
    return True


def _extractive_fallback(documents: list[Document], max_sents: int = 5) -> str:
    """抽取式兜底（无 LLM 或 LLM 失败时）"""
    import re
    
    def split(text: str) -> list[str]:
        return [s for s in re.split(r"[。！？!?；;]\s*", (text or "")) if s]
    
    sents = []
    for doc in documents:
        sents += split(doc.page_content)
        if len(sents) >= max_sents:
            break
    
    if not sents:
        return "根据现有资料无法给出确定答案。"
    
    body = "；".join(sents[:max_sents]) + "。"
    refs = "参考：" + "".join(f"[S{i+1}]" for i in range(len(documents))) if documents else ""
    return body + ("\n\n" + refs if refs else "")


def _postprocess_answer(text: str, documents: list[Document]) -> str:
    """后处理答案，添加引用"""
    text = (text or "").strip()
    refs = "参考：" + "".join(f"[S{i+1}]" for i in range(len(documents))) if documents else ""
    if text and "参考：" not in text and refs:
        text += "\n\n" + refs
    return text


def _no_kb_response(intent_label: str) -> str:
    """非 KB 意图的友好答复"""
    if intent_label in ("闲聊其他", "系统操作"):
        return "这是智慧交通助手。如果你想查询限行规则、信号配时、公交换乘、停车计费或充电规范，请描述更具体的道路/线路/时段/场景。"
    return "该问题暂未接入知识库。请补充更具体的信息（如道路/路段、时间段、线路名），或改问法规/配时/公交/停车/充电/车路协同等主题。"


def create_rag_chain(
    kb_name: Optional[str] = None,
    device: Optional[str] = None,
    use_bm25: bool = USE_BM25,
    use_reranker: bool = USE_RERANKER,
    top_k_final: int = TOP_K_FINAL,
    conf_th: float = CONF_TH,
) -> RunnablePassthrough:
    """
    创建 LangChain RAG Chain
    
    Args:
        kb_name: 知识库名称（如果指定，使用固定知识库；否则使用意图感知检索）
        device: 设备类型
        use_bm25: 是否使用 BM25
        use_reranker: 是否使用重排器
        top_k_final: 最终返回文档数
        conf_th: 意图置信度阈值
    
    Returns:
        LangChain Chain
    """
    # 选择检索器
    if kb_name:
        retriever = CustomRetriever(
            kb_name=kb_name,
            device=device,
            use_bm25=use_bm25,
            use_reranker=use_reranker,
            top_k_final=top_k_final,
        )
    else:
        retriever = IntentAwareRetriever(
            device=device,
            use_bm25=use_bm25,
            use_reranker=use_reranker,
            top_k_final=top_k_final,
            conf_th=conf_th,
        )
    
    # 创建 LLM
    llm = get_langchain_llm()
    
    # 创建 Prompt
    prompt = create_rag_prompt()
    
    # 构建 Chain
    def retrieve_docs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """检索文档（直接调用 modules，不使用 adapter）"""
        # 从输入中提取 query（可能是字符串或字典）
        if isinstance(inputs, dict):
            query = inputs.get("query", "")
        else:
            query = str(inputs)
        
        # 意图识别（使用统一的适配器）
        intent_label, conf, topk_raw = predict_intent(query)
        intent = IntentResult(
            label=intent_label,
            score=conf,
            topk=[IntentTop(l, s) for l, s in (topk_raw or [])]
        )
        
        # 检索
        documents = retriever.get_relevant_documents(query)
        
        # 判断质量
        quality_ok = _hits_quality_ok(documents)
        
        return {
            "query": query,
            "intent": intent,
            "documents": documents,
            "quality_ok": quality_ok,
        }
    
    def format_docs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """格式化文档为上下文"""
        query = inputs["query"]
        documents = inputs.get("documents", [])
        quality_ok = inputs.get("quality_ok", False)
        
        # 如果质量不合格，返回空文档
        if not quality_ok:
            documents = []
        
        context = format_context(documents)
        
        return {
            "query": query,
            "context": context,
            "documents": documents,
            "quality_ok": quality_ok,
            **{k: v for k, v in inputs.items() if k not in ["query", "documents", "quality_ok"]}
        }
    
    def generate_answer(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """生成答案"""
        query = inputs["query"]
        context = inputs["context"]
        documents = inputs.get("documents", [])
        intent = inputs.get("intent")
        quality_ok = inputs.get("quality_ok", False)
        
        # 如果没有文档或质量不合格
        if not documents or not quality_ok:
            intent_label = intent.label if intent else "未知"
            answer = "知识库里没有找到足够可靠的资料。请补充更具体的信息（道路/路段、时间段、线路名），我再查一次；或改问法规/配时/公交/停车/充电/车路协同等主题。"
            if not intent or intent.label not in ["交通法规", "信号配时", "公交规则", "停车政策", "充电规范", "车路协同"]:
                answer = _no_kb_response(intent_label)
            
            return {
                "query": query,
                "answer": answer,
                "documents": [],
                "intent": intent,
                "quality_ok": False,
            }
        
        # 使用 LLM 生成答案
        try:
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"query": query, "context": context})
        except Exception as e:
            # 兜底：抽取式回答
            answer = _extractive_fallback(documents)
        
        # 后处理
        answer = _postprocess_answer(answer, documents)
        
        # 转换为 Hit 列表
        hits = [
            Hit(
                text=doc.page_content,
                score=float(doc.metadata.get("score", 0.0)),
                source=doc.metadata.get("source", ""),
                doc_id=doc.metadata.get("doc_id"),
                chunk_id=doc.metadata.get("chunk_id"),
            )
            for doc in documents
        ]
        
        # 获取 KB 名称
        kb = documents[0].metadata.get("kb_name") if documents else None
        
        return {
            "query": query,
            "answer": answer,
            "documents": documents,
            "hits": hits,
            "intent": intent,
            "kb": kb,
            "quality_ok": True,
        }
    
    # 组装 Chain
    chain = (
        RunnablePassthrough()
        | RunnableLambda(retrieve_docs)
        | RunnableLambda(format_docs)
        | RunnableLambda(generate_answer)
    )
    
    return chain


def rag_answer_langchain(
    query: str,
    *,
    kb_name: Optional[str] = None,
    device: Optional[str] = None,
    conf_th: float = CONF_TH,
    top_k_final: int = TOP_K_FINAL,
    use_bm25: bool = USE_BM25,
    use_reranker: bool = USE_RERANKER,
) -> AnswerPack:
    """
    基于 LangChain 的 RAG 回答接口（兼容原有接口）
    
    Args:
        query: 用户查询
        kb_name: 知识库名称（可选）
        device: 设备类型（可选）
        conf_th: 意图置信度阈值
        top_k_final: 最终返回文档数
        use_bm25: 是否使用 BM25
        use_reranker: 是否使用重排器
    
    Returns:
        AnswerPack（兼容原有格式）
    """
    # 创建 Chain
    chain = create_rag_chain(
        kb_name=kb_name,
        device=device,
        use_bm25=use_bm25,
        use_reranker=use_reranker,
        top_k_final=top_k_final,
        conf_th=conf_th,
    )
    
    # 执行 Chain
    result = chain.invoke({"query": query})
    
    # 构建 AnswerPack
    intent = result.get("intent") or IntentResult(
        label="未知",
        score=0.0,
        topk=[]
    )
    
    hits = result.get("hits", [])
    answer = result.get("answer", "")
    kb = result.get("kb")
    quality_ok = result.get("quality_ok", False)
    
    metrics = Metrics(
        used_kb=kb,
        intent_conf=intent.score,
        conf_th=conf_th,
        notes={"langchain": True, "quality_ok": quality_ok}
    )
    
    return AnswerPack(
        intent=intent,
        kb=kb,
        hits=hits,
        answer=answer,
        metrics=metrics,
    )

