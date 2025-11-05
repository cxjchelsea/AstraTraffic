# -*- coding: utf-8 -*-
"""
核心编排模块
职责：整个系统的编排和流程控制
包含：
- rag_chain: RAG Chain 的组装和编排
"""
from core.rag_chain import (
    create_rag_chain,
    create_rag_chain_with_history,
    rag_answer_langchain,
    rag_answer_with_history,
)

# 为了向后兼容，提供rag_answer别名
rag_answer = rag_answer_langchain

__all__ = [
    "create_rag_chain",
    "create_rag_chain_with_history",
    "rag_answer",
    "rag_answer_langchain",
    "rag_answer_with_history",
]

