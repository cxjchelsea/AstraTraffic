# -*- coding: utf-8 -*-
"""
LangChain Prompt 适配器
调用底层 modules/generator/prompt 实现，适配为 LangChain 格式
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# 导入底层实现
from modules.generator.prompt import RAG_PROMPT_TEMPLATE, format_hits_to_context


def create_rag_prompt() -> PromptTemplate:
    """
    创建 LangChain PromptTemplate（适配层）
    使用底层 modules/generator/prompt 的模板文本
    """
    return PromptTemplate(
        input_variables=["query", "context"],
        template=RAG_PROMPT_TEMPLATE,
    )


def format_context(documents: list[Document]) -> str:
    """
    将 LangChain Document 列表转换为上下文字符串（适配层）
    
    Args:
        documents: LangChain Document 列表
    
    Returns:
        格式化后的上下文字符串
    """
    if not documents:
        return "（无检索片段）"
    
    # 将 Document 转换为字典格式，以便复用底层格式化逻辑
    hits = []
    for doc in documents:
        hits.append({
            "text": doc.page_content or "",
            "source": doc.metadata.get("source", ""),
        })
    
    # 调用底层格式化函数
    return format_hits_to_context(hits)
