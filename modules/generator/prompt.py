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
5) 如果资料中包含路径规划信息（包含"【路径规划】"标记），必须严格按照资料中的信息回答，并明确说明：
   - 出行方式（必须说明是"驾车"、"步行"、"骑行"还是"公交"）
   - 距离必须说明是"实际路径距离"（不是直线距离）
   - 耗时必须明确说明对应的出行方式（例如："驾车预计时间约XX分钟"或"步行预计时间约XX分钟"）
   示例：如果资料显示"出行方式：驾车 | 实际路径距离：8.95公里 | 驾车预计时间：54.5分钟"，则回答应为"从天安门到西直门驾车出行的实际路径距离为8.95公里，驾车预计时间约54.5分钟"
6) 结尾列出引用：格式"参考：[S1][S2]…"。

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

工具选择规则（按优先级顺序判断）：

1. **route_planning（路径规划）**：当用户询问如何从A地到B地、路线规划、怎么走等问题时使用，例如：
   - "从北京站到天安门怎么走"
   - "帮我规划从中关村到西单的路线"
   - "从天安门到故宫"
   - "从这儿到中关村怎么走"
   - "怎么去天安门"
   - 关键词：从...到...、怎么走、路线、路径、规划

2. **realtime_traffic（路况查询）**：当用户询问某条道路的实时拥堵、路况、是否堵车等问题时使用，例如：
   - "中关村大街的路况"
   - "三环路拥堵情况"
   - "这条路堵吗"
   - "长安街现在怎么样"
   - 关键词：路况、拥堵、堵车、堵吗、怎么样（针对道路）

3. **realtime_map（地图查看）**：当用户明确询问某个具体地点、位置、区域的地图时使用，例如：
   - "查看中关村的地图"
   - "天安门在哪里"
   - "显示北京市朝阳区的地图"
   - "天安门地图"
   - 关键词：地图、在哪里、位置、查看...地图

4. **kb_law（交通法规）**：当用户询问交通法规、限行、处罚、罚款、记分等问题时使用，例如：
   - "限行怎么处罚"
   - "闯红灯扣几分"
   - "违停罚款多少"
   - "限行规定"
   - 关键词：限行、处罚、罚款、扣分、记分、法规、规定

5. **kb_ev（充电规范）**：当用户询问电动汽车、充电、电桩等问题时使用，例如：
   - "电车充电"
   - "充电桩在哪里"
   - "电动汽车充电规范"
   - 关键词：充电、电桩、电动车、电动汽车

6. **kb_parking（停车政策）**：当用户询问停车、泊位等问题时使用，例如：
   - "停车收费标准"
   - "哪里可以停车"
   - 关键词：停车、泊位

7. **kb_transit（公交规则）**：当用户询问公交、地铁、换乘等问题时使用，例如：
   - "公交线路"
   - "地铁换乘"
   - 关键词：公交、地铁、换乘

8. **kb_handbook（信号配时）**：当用户询问信号配时、诱导策略等技术性问题时使用，例如：
   - "信号灯配时"
   - "诱导策略"
   - 关键词：信号、配时、诱导

9. **kb_iov（车路协同）**：当用户询问车路协同、V2X等问题时使用

10. **kb_health（健康咨询）**：当用户询问健康、医疗等问题时使用

11. **kb_report（报告查询）**：当用户询问报告、分析等问题时使用

12. **none（日常对话）**：用于问候语、简单对话、无需知识库的普通交流，例如：
    - "你好"、"再见"、"谢谢"、"好的"、"知道了"
    - 简短的确认或回应（"嗯"、"哦"、"明白"等）

**重要判断原则**：
- 优先判断是否为路径规划（包含"从...到..."、"怎么走"等）
- 其次判断是否为路况查询（询问道路拥堵情况）
- 再次判断是否为地图查看（询问地点位置）
- 最后判断是否为知识问答（选择相应的知识库工具）

用户查询：{query}

选择的工具（只返回工具ID，如：route_planning、kb_law、realtime_traffic等）："""


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


# ==================== 起点终点提取 Prompt ====================

ROUTE_EXTRACTION_PROMPT_TEMPLATE = """请从以下用户查询中提取路径规划的起点和终点。

重要规则：
1. 如果查询不包含明确的起点和终点，请返回"无"
2. 起点和终点可以是地点名称、地址或坐标
3. 输出格式：起点|终点（用"|"分隔，如果缺少某个则用"无"代替）

示例：
- 查询："从北京站到天安门怎么走" → 输出：北京站|天安门
- 查询："帮我规划从中关村到西单的路线" → 输出：中关村|西单
- 查询："从天安门到故宫" → 输出：天安门|故宫
- 查询："怎么去天安门" → 输出：无|天安门（如果上下文没有起点信息）
- 查询："从我家到公司" → 输出：我家|公司（如果上下文有这些信息）
- 查询："路径规划" → 输出：无
- 查询："你好" → 输出：无

用户查询：{query}

提取的起点和终点（格式：起点|终点）："""


def build_route_extraction_prompt(query: str) -> str:
    """
    构建起点终点提取的prompt

    Args:
        query: 用户查询

    Returns:
        完整的prompt字符串
    """
    return ROUTE_EXTRACTION_PROMPT_TEMPLATE.format(query=query)


# ==================== 出行方式提取 Prompt ====================

ROUTE_TYPE_EXTRACTION_PROMPT_TEMPLATE = """请从以下用户查询中提取出行方式。

重要规则：
1. 如果查询明确提到出行方式，请返回对应的出行方式
2. 如果查询没有明确提到，请返回"驾车"（默认）
3. 输出格式：只返回一个词（驾车/步行/骑行/公交）

出行方式对应关系：
- "驾车"、"开车"、"自驾" → 驾车
- "步行"、"走路"、"徒步" → 步行
- "骑行"、"骑车"、"自行车" → 骑行
- "公交"、"地铁"、"公共交通" → 公交

示例：
- 查询："从天安门到故宫怎么走" → 输出：驾车（默认）
- 查询："天安门到故宫步行怎么走" → 输出：步行
- 查询："从天安门到故宫骑车怎么走" → 输出：骑行
- 查询："从天安门到故宫坐公交怎么走" → 输出：公交
- 查询："从天安门到故宫开车怎么走" → 输出：驾车

用户查询：{query}

提取的出行方式（只返回一个词）："""


def build_route_type_extraction_prompt(query: str) -> str:
    """
    构建出行方式提取的prompt

    Args:
        query: 用户查询

    Returns:
        完整的prompt字符串
    """
    return ROUTE_TYPE_EXTRACTION_PROMPT_TEMPLATE.format(query=query)

