# -*- coding: utf-8 -*-
"""
Prompt 模板（底层实现，不依赖 LangChain）
包含交通领域 RAG 的提示词模板和通用格式化逻辑
"""
from typing import List, Union, Dict, Any


# RAG Prompt 模板文本（业务逻辑，不依赖任何框架）
RAG_PROMPT_TEMPLATE = """你是严谨的中文智慧交通助手。仅依据"资料片段"回答；如果资料不足，请明确说"根据现有资料无法给出确定答案"，不要编造。

# 问题
{query}

# 资料片段（可能不完整）
{context}

# 输出要求（必须遵守）
1) 先给出【核心结论】（3–5句，简洁明确）；
2) 然后给出【建议/步骤】（如需查询具体路段/线路/时段，应提示用户补充信息；若问题涉及实时数据，请提醒以官方/平台实时公告为准）；
3) 若问题涉及法规与处罚，用谨慎语气并提示"以当地官方发布为准"；
4) 不得杜撰资料外的信息；信息不足要直说；
5) 结尾列出引用：格式"参考：[S1][S2]…"。

# 输出格式（照抄并填充）
- 核心结论：
- 建议/步骤：
- 参考：[S1] [S2] …
"""


def format_hits_to_context(hits: List) -> str:
    """
    将检索结果（Hit 列表）格式化为上下文字符串（通用实现）
    
    Args:
        hits: Hit 对象列表或包含 text 和 source 的字典列表
    
    Returns:
        格式化后的上下文字符串
    """
    if not hits:
        return "（无检索片段）"
    
    ctx_lines = []
    for i, hit in enumerate(hits, 1):
        # 兼容 Hit 对象或字典
        if hasattr(hit, 'text'):
            text = (hit.text or "").replace("\n", " ").strip()
            source = (hit.source or "").strip()
        elif isinstance(hit, dict):
            text = (hit.get("text") or "").replace("\n", " ").strip()
            source = (hit.get("source") or "").strip()
        else:
            text = str(hit).replace("\n", " ").strip()
            source = ""
        
        ctx_lines.append(f"[S{i}] {text} (source: {source})")
    
    return "\n".join(ctx_lines)


def build_prompt(query: str, hits: List) -> str:
    """
    构建完整的 prompt 字符串（通用实现，不依赖框架）
    
    Args:
        query: 用户查询
        hits: Hit 对象列表或包含 text 和 source 的字典列表
    
    Returns:
        完整的 prompt 字符串
    """
    context = format_hits_to_context(hits)
    return RAG_PROMPT_TEMPLATE.format(query=query, context=context)

