# -*- coding: utf-8 -*-
"""
适配器模块（LangChain接口适配）
职责：将底层实现适配为 LangChain 接口
包含：
- retriever: 检索适配器
- llm: LLM适配器
- prompt: Prompt适配器
- query_rewriter: 查询改写适配器
"""
from adapters.retriever import (
    CustomRetriever,
    ToolAwareRetriever,
    retrieve,
    get_searcher,
)
from adapters.llm import (
    get_langchain_llm,
    get_llm_client,
    CustomLLM,
)
from adapters.prompt import (
    create_rag_prompt,
    create_rag_prompt_with_history,
    format_context,
)
from adapters.query_rewriter import rewrite_query_with_history

__all__ = [
    "CustomRetriever",
    "ToolAwareRetriever",
    "retrieve",
    "get_searcher",
    "get_langchain_llm",
    "get_llm_client",
    "CustomLLM",
    "create_rag_prompt",
    "create_rag_prompt_with_history",
    "format_context",
    "rewrite_query_with_history",
]


