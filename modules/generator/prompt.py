# -*- coding: utf-8 -*-
"""
Prompt 模板（底层实现，不依赖 LangChain）
职责：管理所有发送给LLM的prompt模板（输入给LLM的内容）
包含：
- RAG问答的prompt模板
- Tool选择的prompt模板
- 道路名称提取的prompt模板
- 查询改写的prompt模板（在query_rewriter.py中）
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


# ==================== 简单对话 Prompt ====================

# 简单对话 Prompt（用于问候语、日常对话等，不需要知识库）
CHAT_PROMPT_TEMPLATE = """你是一个友好的中文智慧交通助手。请用自然、友好的方式回答用户的问题。

用户说：{query}

请用简洁、友好的方式回答（1-2句话即可）。如果用户是问候，请礼貌地回应并询问是否需要帮助。"""


def build_chat_prompt(query: str) -> str:
    """
    构建简单对话的prompt（用于问候语等，不需要知识库）
    
    Args:
        query: 用户查询
    
    Returns:
        完整的prompt字符串
    """
    return CHAT_PROMPT_TEMPLATE.format(query=query)


# ==================== 多轮对话支持 ====================

# RAG Prompt 模板（支持历史对话）
RAG_PROMPT_WITH_HISTORY_TEMPLATE = """你是严谨的中文智慧交通助手。仅依据"资料片段"回答；如果资料不足，请明确说"根据现有资料无法给出确定答案"，不要编造。

{history_section}

# 当前问题
{query}

# 资料片段（可能不完整）
{context}

# 输出要求（必须遵守）
1) 先给出【核心结论】（3–5句，简洁明确）；
2) 然后给出【建议/步骤】（如需查询具体路段/线路/时段，应提示用户补充信息；若问题涉及实时数据，请提醒以官方/平台实时公告为准）；
3) 若问题涉及法规与处罚，用谨慎语气并提示"以当地官方发布为准"；
4) 考虑对话历史，保持回答的连贯性和上下文一致性；
5) 不得杜撰资料外的信息；信息不足要直说；
6) 结尾列出引用：格式"参考：[S1][S2]…"。

# 输出格式（照抄并填充）
- 核心结论：
- 建议/步骤：
- 参考：[S1] [S2] …
"""


def format_chat_history(history: List[tuple]) -> str:
    """
    格式化对话历史用于Prompt
    
    Args:
        history: 对话历史列表，格式为 [(用户问题, 助手回答), ...]
    
    Returns:
        格式化后的历史文本，如果没有历史则返回空字符串
    """
    if not history:
        return ""
    
    lines = ["# 对话历史（最近的{}轮）".format(len(history))]
    for i, (user_q, assistant_a) in enumerate(history, 1):
        # 截断过长的回答（只保留前150字）
        truncated_answer = assistant_a[:150] + "..." if len(assistant_a) > 150 else assistant_a
        lines.append(f"轮次{i}:")
        lines.append(f"  用户：{user_q}")
        lines.append(f"  助手：{truncated_answer}")
    
    return "\n".join(lines) + "\n"


def build_prompt_with_history(query: str, hits: List, history: List[tuple] = None) -> str:
    """
    构建包含对话历史的完整 prompt 字符串
    
    Args:
        query: 当前用户查询
        hits: Hit 对象列表或包含 text 和 source 的字典列表
        history: 对话历史列表，格式为 [(用户问题, 助手回答), ...]，可选
    
    Returns:
        完整的 prompt 字符串
    """
    context = format_hits_to_context(hits)
    history_text = format_chat_history(history) if history else ""
    
    if history_text:
        history_section = history_text
    else:
        history_section = ""
    
    return RAG_PROMPT_WITH_HISTORY_TEMPLATE.format(
        history_section=history_section,
        query=query,
        context=context
    )


# ==================== Tool选择器Prompt ====================

TOOL_SELECTION_PROMPT_TEMPLATE = """可用工具：
{tools_description}

工具选择规则：
1. **none（日常对话）**：用于问候语、简单对话、无需知识库的普通交流，例如：
   - "你好"、"再见"、"谢谢"、"好的"、"知道了"、"早上好"、"今天天气如何"
   - 简短的确认或回应（"嗯"、"哦"、"明白"等）
   - 这些查询不需要检索知识库，也不需要调用实时工具，直接由LLM友好回答

2. **realtime_map（地图查看）**：仅当用户明确询问某个具体地点、位置、区域的地图时使用，例如：
   - "查看中关村的地图"
   - "天安门在哪里"
   - "显示北京市朝阳区的地图"
   - "大兴区地图"

3. **realtime_traffic（路况查询）**：仅当用户明确询问某条道路或区域的实时路况时使用，例如：
   - "中关村大街的路况"
   - "三环路拥堵情况"

4. **知识库工具（kb_*）**：用于交通法规、政策、规则等知识问答，或当查询不涉及实时数据时使用
   - 查询交通相关问题（限行、处罚、停车、公交等）→ 选择相应的知识库工具

5. **选择原则**：
   - 查询是问候语或简单对话 → none
   - 查询中包含明确的地点名称且用户想查看地图 → realtime_map
   - 查询中包含明确的道路名称且询问路况 → realtime_traffic
   - 其他知识问答 → 选择相应的知识库工具（kb_*）

用户查询：{query}

选择的工具："""


def build_tool_selection_prompt(query: str, tool_descriptions: List[tuple]) -> str:
    """
    构建Tool选择器的prompt
    
    Args:
        query: 用户查询
        tool_descriptions: 工具描述列表，格式为 [(工具名称, 工具描述), ...]
    
    Returns:
        完整的prompt字符串
    """
    tools_lines = []
    for i, (tool_name, tool_desc) in enumerate(tool_descriptions, 1):
        tools_lines.append(f"{i}. {tool_name} - {tool_desc}")
    
    tools_description = "\n".join(tools_lines)
    
    return TOOL_SELECTION_PROMPT_TEMPLATE.format(
        tools_description=tools_description,
        query=query
    )


# ==================== 道路名称提取Prompt ====================

ROAD_NAME_EXTRACTION_PROMPT_TEMPLATE = """从以下用户查询中提取道路名称。只返回道路名称本身，不要包含其他内容（如"的路况"、"怎么样"等）。

如果查询中没有明确提及道路名称，请返回"无"。

示例：
- 查询："中关村大街的路况怎么样？" → 输出：中关村大街
- 查询："三环路拥堵情况" → 输出：三环路
- 查询："长安街现在怎么样" → 输出：长安街
- 查询："北京现在哪些地方堵车？" → 输出：无

用户查询：{query}

提取的道路名称："""


def build_road_name_extraction_prompt(query: str) -> str:
    """
    构建道路名称提取的prompt
    
    Args:
        query: 用户查询
    
    Returns:
        完整的prompt字符串
    """
    return ROAD_NAME_EXTRACTION_PROMPT_TEMPLATE.format(query=query)


# ==================== 地点名称提取 Prompt ====================

LOCATION_EXTRACTION_PROMPT_TEMPLATE = """请从以下用户查询中提取地点名称（如城市、区域、建筑物、景点等）。

重要规则：
1. 如果查询是问候语、普通对话、或没有明确的地名，请返回"无"
2. 常见问候语（如"你好"、"再见"、"谢谢"等）不应被提取为地名
3. 只有明确包含地点名称的查询才提取

示例：
- 查询："查看中关村的地图" → 输出：中关村
- 查询："天安门在哪里" → 输出：天安门
- 查询："北京市朝阳区的地图" → 输出：北京市朝阳区
- 查询："显示地图" → 输出：无
- 查询："你好" → 输出：无
- 查询："再见" → 输出：无
- 查询："谢谢" → 输出：无

用户查询：{query}

提取的地点名称："""


def build_location_extraction_prompt(query: str) -> str:
    """
    构建地点名称提取的prompt
    
    Args:
        query: 用户查询
    
    Returns:
        完整的prompt字符串
    """
    return LOCATION_EXTRACTION_PROMPT_TEMPLATE.format(query=query)

