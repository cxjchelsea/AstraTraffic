# -*- coding: utf-8 -*-
"""
核心 RAG Chain 编排
使用 LangChain 的 LCEL (LangChain Expression Language) 构建 RAG 流程

业务层归属：编排层（协调五层协作）
- 协调感知层（L1）：调用实时数据获取
- 协调理解层（L2）：调用查询改写、上下文理解
- 协调决策层（L3）：调用工具选择、质量评估
- 协调执行层（L4）：调用知识检索、工具执行
- 未来：协调反思层（L5）：学习与优化

当前模式：被动响应式（用户查询触发）
未来演进：主动自主式（持续感知 → 主动决策 → 自主行动）
"""
from typing import Optional, Dict, Any, Callable
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# 适配器导入
from adapters.retriever import CustomRetriever, ToolAwareRetriever
from adapters.llm import get_langchain_llm
from adapters.prompt import create_rag_prompt, format_context, create_rag_prompt_with_history

# 业务逻辑导入
from services.quality import check_hits_quality
from services.fallback import (
    get_no_document_response,
    should_fallback,
    check_has_realtime_info,
    extractive_fallback,
)
from services.tool_selector import select_tool
from services.realtime import execute_realtime_tool
from services.input_handler import get_history_manager

# 底层实现导入
from modules.config.tool_config import tool_to_intent_label
from modules.rag_types.rag_types import AnswerPack, IntentResult, IntentTop, Metrics, Hit
from modules.config.settings import USE_BM25, USE_RERANKER, TOP_K_FINAL, CONF_TH, USE_TOOL_SELECTOR


# ==================== 工具函数 ====================

def _postprocess_answer(text: str, documents: list[Document], tool_selection=None) -> str:
    """
    后处理答案，添加引用
    
    Args:
        text: LLM生成的答案文本
        documents: 检索到的文档列表
        tool_selection: 工具选择结果（可选），用于判断是否为实时工具查询
    """
    text = (text or "").strip()
    
    # 如果是实时工具查询，移除LLM生成的引用标记（实时工具查询不依赖知识库文档）
    if tool_selection and tool_selection.tool in ["realtime_traffic", "realtime_map"]:
        # 移除可能存在的引用标记行（如 "- 参考：[S1] [S2] …" 或 "参考：[S1]"）
        import re
        # 匹配以"-"、"- "或"参考："开头，后面跟着 [S数字] 的整行（包括前后换行符）
        # 匹配格式：- 参考：[S1] 或 - 参考：[S1] [S2] 或 参考：[S1] 等
        # 使用多行模式，匹配整行
        lines = text.split('\n')
        filtered_lines = []
        for line in lines:
            # 检查是否是引用标记行
            if re.match(r'^\s*-?\s*参考：\s*\[S\d+\]', line.strip()):
                # 跳过引用标记行
                continue
            filtered_lines.append(line)
        text = '\n'.join(filtered_lines)
        # 清理多余的换行符
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    # 如果有文档，添加引用标记
    refs = "参考：" + "".join(f"[S{i+1}]" for i in range(len(documents))) if documents else ""
    if text and "参考：" not in text and refs:
        text += "\n\n" + refs
    return text


def _convert_documents_to_hits(documents: list[Document]) -> list[Hit]:
    """将 Document 列表转换为 Hit 列表"""
    return [
        Hit(
            text=doc.page_content,
            score=float(doc.metadata.get("score", 0.0)),
            source=doc.metadata.get("source", ""),
            doc_id=doc.metadata.get("doc_id"),
            chunk_id=doc.metadata.get("chunk_id"),
        )
        for doc in documents
    ]


def _create_intent_from_tool_selection(tool_selection) -> IntentResult:
    """基于tool选择结果创建intent对象（用于兼容性）"""
    intent_label = tool_to_intent_label(tool_selection.tool)
    return IntentResult(
        label=intent_label,
        score=tool_selection.confidence,
        topk=[IntentTop(intent_label, tool_selection.confidence)]
    )


def _create_default_intent() -> IntentResult:
    """创建默认意图对象"""
    return IntentResult(
        label="知识库查询",
        score=1.0,
        topk=[IntentTop("知识库查询", 1.0)]
    )


def _extract_query_from_inputs(inputs: Dict[str, Any]) -> str:
    """从输入中提取查询文本"""
    if isinstance(inputs, dict):
        return inputs.get("query", "")
    return str(inputs)


# ==================== Chain 构建函数 ====================

def _build_retrieve_docs_fn(
    retriever: CustomRetriever | ToolAwareRetriever,
    kb_name: Optional[str],
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """构建检索文档的函数"""
    def retrieve_docs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """检索文档"""
        query = _extract_query_from_inputs(inputs)
        
        # 使用tool选择器（仅LLM模式）
        tool_selection = None
        if not kb_name:
            tool_selection = select_tool(query)
            intent = _create_intent_from_tool_selection(tool_selection)
        else:
            intent = _create_default_intent()
        
        # 如果工具是none（日常对话），跳过检索，直接返回空文档
        if tool_selection and tool_selection.tool == "none":
            return {
                "query": query,
                "intent": intent,
                "documents": [],
                "quality_ok": False,
                "tool_selection": tool_selection,
                **{k: v for k, v in inputs.items() if k not in ["query"]}
            }
        
        # 检索
        documents = retriever.get_relevant_documents(query)
        quality_ok = check_hits_quality(documents)
        
        return {
            "query": query,
            "intent": intent,
            "documents": documents,
            "quality_ok": quality_ok,
            "tool_selection": tool_selection,
            **{k: v for k, v in inputs.items() if k not in ["query"]}
        }
    
    return retrieve_docs


def _build_retrieve_docs_with_history_fn(
    retriever: CustomRetriever | ToolAwareRetriever,
    kb_name: Optional[str],
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """构建支持历史的检索文档函数"""
    def retrieve_docs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """检索文档（支持查询改写）"""
        query = inputs.get("query", "")
        history = inputs.get("history", [])
        
        # 查询改写
        from adapters.query_rewriter import rewrite_query_with_history
        original_query = query
        rewritten_query = rewrite_query_with_history(query, history, max_history_turns=3) if history else query
        
        # 使用tool选择器（仅LLM模式）
        tool_selection = None
        if not kb_name:
            tool_selection = select_tool(rewritten_query)
            intent = _create_intent_from_tool_selection(tool_selection)
        else:
            intent = _create_default_intent()
        
        # 如果工具是none（日常对话），跳过检索，直接返回空文档
        if tool_selection and tool_selection.tool == "none":
            return {
                "query": original_query,
                "rewritten_query": rewritten_query,
                "intent": intent,
                "documents": [],
                "quality_ok": False,
                "tool_selection": tool_selection,
                "history": history,
                **{k: v for k, v in inputs.items() if k not in ["query"]}
            }
        
        # 检索（使用改写后的查询）
        documents = retriever.get_relevant_documents(rewritten_query)
        quality_ok = check_hits_quality(documents)
        
        return {
            "query": original_query,
            "rewritten_query": rewritten_query,
            "intent": intent,
            "documents": documents,
            "quality_ok": quality_ok,
            "history": history,
            "tool_selection": tool_selection,
        }
    
    return retrieve_docs


def _build_format_docs_fn() -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """构建格式化文档的函数"""
    def format_docs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """格式化文档为上下文"""
        query = inputs["query"]
        documents = inputs.get("documents", [])
        quality_ok = inputs.get("quality_ok", False)
        
        # 如果质量不合格，返回空文档
        if not quality_ok:
            documents = []
        
        context = format_context(documents)
        
        # 使用实时工具执行器统一处理实时API工具（跳过none工具）
        tool_selection = inputs.get("tool_selection")
        realtime_result = None
        if tool_selection and tool_selection.tool not in ["none"] and tool_selection.tool in ["realtime_traffic", "realtime_map", "route_planning"]:
            realtime_result = execute_realtime_tool(tool_selection, query)
            if realtime_result:
                context = realtime_result.context_text + ("\n\n" + context if context else "")
        
        result = {
            "query": query,
            "context": context,
            "documents": documents,
            "quality_ok": quality_ok,
        }
        
        # 保留其他字段（如 rewritten_query, history 等）
        for key in ["rewritten_query", "history", "intent", "tool_selection"]:
            if key in inputs:
                result[key] = inputs[key]
        
        # 保存实时工具结果（包含地图数据）
        if realtime_result:
            result["realtime_result"] = realtime_result
        
        return result
    
    return format_docs


def _build_format_docs_with_history_fn() -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """构建支持历史的格式化文档函数"""
    def format_docs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """格式化文档为上下文（支持历史对话）"""
        query = inputs["query"]
        documents = inputs.get("documents", [])
        quality_ok = inputs.get("quality_ok", False)
        history = inputs.get("history", [])
        
        # 如果质量不合格，返回空文档
        if not quality_ok:
            documents = []
        
        context = format_context(documents)
        
        # 使用实时工具执行器统一处理实时API工具（跳过none工具）
        tool_selection = inputs.get("tool_selection")
        realtime_result = None
        if tool_selection and tool_selection.tool not in ["none"] and tool_selection.tool in ["realtime_traffic", "realtime_map", "route_planning"]:
            realtime_result = execute_realtime_tool(tool_selection, query)
            if realtime_result:
                context = realtime_result.context_text + ("\n\n" + context if context else "")
        
        # 格式化历史对话
        from modules.generator.prompt import format_chat_history
        history_text = format_chat_history(history) if history else ""
        
        result = {
            "query": query,
            "rewritten_query": inputs.get("rewritten_query", query),
            "context": context,
            "documents": documents,
            "quality_ok": quality_ok,
            "history": history,
            "history_text": history_text,
            "intent": inputs.get("intent"),
            "tool_selection": inputs.get("tool_selection"),
        }
        
        # 保存实时工具结果（包含地图数据）
        if realtime_result:
            result["realtime_result"] = realtime_result
        
        return result
    
    return format_docs


def _build_generate_answer_fn(
    llm,
    prompt,
    with_history: bool = False,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """构建生成答案的函数"""
    def generate_answer(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """生成答案"""
        query = inputs["query"]
        context = inputs["context"]
        documents = inputs.get("documents", [])
        intent = inputs.get("intent")
        quality_ok = inputs.get("quality_ok", False)
        
        # 如果工具是none（日常对话），使用简单对话prompt直接让LLM回答
        tool_selection = inputs.get("tool_selection")
        if tool_selection and tool_selection.tool == "none":
            try:
                from modules.generator.prompt import build_chat_prompt
                from adapters.llm import get_llm_client
                
                chat_prompt = build_chat_prompt(query)
                llm_client = get_llm_client()
                answer = llm_client(chat_prompt, temperature=0.7, max_tokens=100)
                
                result = {
                    "query": query,
                    "answer": answer.strip(),
                    "documents": [],
                    "intent": intent,
                    "quality_ok": False,
                }
                if with_history:
                    result["rewritten_query"] = inputs.get("rewritten_query", query)
                    result["history"] = inputs.get("history", [])
                return result
            except Exception as e:
                # 如果LLM调用失败，使用默认回复
                answer = "你好！我是智慧交通助手，有什么可以帮助您的吗？"
                result = {
                    "query": query,
                    "answer": answer,
                    "documents": [],
                    "intent": intent,
                    "quality_ok": False,
                }
                if with_history:
                    result["rewritten_query"] = inputs.get("rewritten_query", query)
                    result["history"] = inputs.get("history", [])
                return result
        
        # 如果没有文档或质量不合格，使用fallback适配器处理
        has_realtime_info = check_has_realtime_info(context)
        
        if should_fallback(documents, quality_ok, has_realtime_info):
            answer = get_no_document_response(intent, has_realtime_info)
            result = {
                "query": query,
                "answer": answer,
                "documents": [],
                "intent": intent,
                "quality_ok": False,
            }
            if with_history:
                result["rewritten_query"] = inputs.get("rewritten_query", query)
                result["history"] = inputs.get("history", [])
            return result
        
        # 使用 LLM 生成答案
        try:
            if with_history:
                prompt_input = {
                    "query": query,
                    "context": context,
                    "history": inputs.get("history_text", ""),
                }
            else:
                prompt_input = {"query": query, "context": context}
            
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke(prompt_input)
        except Exception as e:
            # 兜底：抽取式回答（LLM调用失败但有文档时）
            answer = extractive_fallback(documents)
        
        # 后处理（传递tool_selection以判断是否需要添加引用）
        tool_selection = inputs.get("tool_selection")
        answer = _postprocess_answer(answer, documents, tool_selection=tool_selection)
        
        # 转换为 Hit 列表
        hits = _convert_documents_to_hits(documents)
        
        # 获取 KB 名称
        kb = documents[0].metadata.get("kb_name") if documents else None
        
        # 提取地图数据或路径数据（如果存在）
        map_data = None
        route_data = None
        realtime_result = inputs.get("realtime_result")
        if realtime_result and realtime_result.metadata:
            map_data = realtime_result.metadata.get("map_data")
            route_data = realtime_result.metadata.get("route_data")
        
        result = {
            "query": query,
            "answer": answer,
            "documents": documents,
            "hits": hits,
            "intent": intent,
            "kb": kb,
            "quality_ok": True,
        }
        
        # 添加地图数据或路径数据到结果中（将在后续构建Metrics时使用）
        if map_data:
            result["map_data"] = map_data
        if route_data:
            result["route_data"] = route_data
        
        if with_history:
            result["rewritten_query"] = inputs.get("rewritten_query", query)
            result["history"] = inputs.get("history", [])
        
        return result
    
    return generate_answer


# ==================== 公共 Chain 创建函数 ====================

def create_rag_chain(
    kb_name: Optional[str] = None,
    device: Optional[str] = None,
    use_bm25: bool = USE_BM25,
    use_reranker: bool = USE_RERANKER,
    top_k_final: int = TOP_K_FINAL,
    conf_th: float = CONF_TH,
    use_tool_selector: Optional[bool] = None,
) -> RunnablePassthrough:
    """
    创建 LangChain RAG Chain
    
    Args:
        kb_name: 知识库名称（如果指定，使用固定知识库；否则使用tool选择器）
        device: 设备类型
        use_bm25: 是否使用 BM25
        use_reranker: 是否使用重排器
        top_k_final: 最终返回文档数
        conf_th: 意图置信度阈值（已废弃，tool选择器使用LLM）
        use_tool_selector: 是否使用tool选择器（None表示使用配置文件默认值，默认True）
    
    Returns:
        LangChain Chain
    """
    # 确定是否使用tool选择器（默认使用，仅LLM模式）
    _use_tool_selector = use_tool_selector if use_tool_selector is not None else USE_TOOL_SELECTOR
    
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
        retriever = ToolAwareRetriever(
            device=device,
            use_bm25=use_bm25,
            use_reranker=use_reranker,
            top_k_final=top_k_final,
        )
    
    # 创建 LLM 和 Prompt
    llm = get_langchain_llm()
    prompt = create_rag_prompt()
    
    # 构建 Chain 函数
    retrieve_fn = _build_retrieve_docs_fn(retriever, kb_name)
    format_fn = _build_format_docs_fn()
    generate_fn = _build_generate_answer_fn(llm, prompt, with_history=False)
    
    # 组装 Chain
    chain = (
        RunnablePassthrough()
        | RunnableLambda(retrieve_fn)
        | RunnableLambda(format_fn)
        | RunnableLambda(generate_fn)
    )
    
    return chain


def create_rag_chain_with_history(
    kb_name: Optional[str] = None,
    device: Optional[str] = None,
    use_bm25: bool = USE_BM25,
    use_reranker: bool = USE_RERANKER,
    top_k_final: int = TOP_K_FINAL,
    conf_th: float = CONF_TH,
    use_tool_selector: Optional[bool] = None,
) -> RunnablePassthrough:
    """
    创建支持多轮对话的 LangChain RAG Chain
    
    Args:
        kb_name: 知识库名称（如果指定，使用固定知识库；否则使用tool选择器）
        device: 设备类型
        use_bm25: 是否使用 BM25
        use_reranker: 是否使用重排器
        top_k_final: 最终返回文档数
        conf_th: 意图置信度阈值（已废弃，tool选择器使用LLM）
        use_tool_selector: 是否使用tool选择器（None表示使用配置文件默认值，默认True）
    
    Returns:
        LangChain Chain（支持 history 参数）
    """
    # 确定是否使用tool选择器（默认使用，仅LLM模式）
    _use_tool_selector = use_tool_selector if use_tool_selector is not None else USE_TOOL_SELECTOR
    
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
        retriever = ToolAwareRetriever(
            device=device,
            use_bm25=use_bm25,
            use_reranker=use_reranker,
            top_k_final=top_k_final,
        )
    
    # 创建 LLM 和 Prompt
    llm = get_langchain_llm()
    prompt = create_rag_prompt_with_history()
    
    # 构建 Chain 函数
    retrieve_fn = _build_retrieve_docs_with_history_fn(retriever, kb_name)
    format_fn = _build_format_docs_with_history_fn()
    generate_fn = _build_generate_answer_fn(llm, prompt, with_history=True)
    
    # 组装 Chain
    chain = (
        RunnablePassthrough()
        | RunnableLambda(retrieve_fn)
        | RunnableLambda(format_fn)
        | RunnableLambda(generate_fn)
    )
    
    return chain


# ==================== 公共接口函数 ====================

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
    
    metrics = Metrics(
        used_kb=result.get("kb"),
        intent_conf=intent.score,
        conf_th=conf_th,
        notes={"langchain": True, "quality_ok": result.get("quality_ok", False)}
    )
    
    return AnswerPack(
        intent=intent,
        kb=result.get("kb"),
        hits=result.get("hits", []),
        answer=result.get("answer", ""),
        metrics=metrics,
    )


def rag_answer_with_history(
    query: str,
    session_id: str = "default",
    *,
    kb_name: Optional[str] = None,
    device: Optional[str] = None,
    conf_th: float = CONF_TH,
    top_k_final: int = TOP_K_FINAL,
    use_bm25: bool = USE_BM25,
    use_reranker: bool = USE_RERANKER,
    max_history_turns: int = 5,
) -> AnswerPack:
    """
    基于 LangChain 的多轮对话 RAG 回答接口
    
    Args:
        query: 用户查询
        session_id: 会话ID（用于管理对话历史）
        kb_name: 知识库名称（可选）
        device: 设备类型（可选）
        conf_th: 意图置信度阈值
        top_k_final: 最终返回文档数
        use_bm25: 是否使用 BM25
        use_reranker: 是否使用重排器
        max_history_turns: 最多使用多少轮历史对话（默认5轮）
    
    Returns:
        AnswerPack（兼容原有格式）
    """
    # 获取历史管理器
    history_manager = get_history_manager()
    
    # 获取对话历史
    history = history_manager.get_history(session_id, max_turns=max_history_turns)
    
    # 创建 Chain
    chain = create_rag_chain_with_history(
        kb_name=kb_name,
        device=device,
        use_bm25=use_bm25,
        use_reranker=use_reranker,
        top_k_final=top_k_final,
        conf_th=conf_th,
    )
    
    # 执行 Chain（传入历史和查询）
    result = chain.invoke({
        "query": query,
        "history": history,
    })
    
    # 构建 AnswerPack
    intent = result.get("intent") or IntentResult(
        label="未知",
        score=0.0,
        topk=[]
    )
    
    rewritten_query = result.get("rewritten_query", query)
    
    # 构建notes，包含地图数据（如果存在）
    notes = {
        "langchain": True,
        "quality_ok": result.get("quality_ok", False),
        "has_history": len(history) > 0,
        "rewritten_query": rewritten_query if rewritten_query != query else None,
    }
    
    # 添加地图数据或路径数据（如果存在）
    map_data = result.get("map_data")
    route_data = result.get("route_data")
    if map_data:
        notes["map_data"] = map_data
    if route_data:
        notes["route_data"] = route_data
    
    metrics = Metrics(
        used_kb=result.get("kb"),
        intent_conf=intent.score,
        conf_th=conf_th,
        notes=notes
    )
    
    # 保存本轮对话到历史
    history_manager.add_exchange(session_id, query, result.get("answer", ""))
    
    return AnswerPack(
        intent=intent,
        kb=result.get("kb"),
        hits=result.get("hits", []),
        answer=result.get("answer", ""),
        metrics=metrics,
    )
