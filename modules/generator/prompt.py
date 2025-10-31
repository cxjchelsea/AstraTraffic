from typing import List
from process.types import Hit

def build_prompt(query: str, hits: List[Hit]) -> str:
    """
    交通领域 RAG 提示词（保留原函数签名，直接替换即可）。
    约束：仅依据“资料片段”作答；信息不足时要明确说明。
    输出结构统一，便于前端渲染与人工复核。
    """
    ctx_lines = []
    for i, h in enumerate(hits, 1):
        text = (h.text or "").replace("\n", " ").strip()
        src = (h.source or "").strip()
        ctx_lines.append(f"[S{i}] {text} (source: {src})")
    ctx_block = "\n".join(ctx_lines) if ctx_lines else "（无检索片段）"

    return f"""你是严谨的中文智慧交通助手。仅依据“资料片段”回答；如果资料不足，请明确说“根据现有资料无法给出确定答案”，不要编造。

# 问题
{query}

# 资料片段（可能不完整）
{ctx_block}

# 输出要求（必须遵守）
1) 先给出【核心结论】（3–5句，简洁明确）；
2) 然后给出【建议/步骤】（如需查询具体路段/线路/时段，应提示用户补充信息；若问题涉及实时数据，请提醒以官方/平台实时公告为准）；
3) 若问题涉及法规与处罚，用谨慎语气并提示“以当地官方发布为准”；
4) 不得杜撰资料外的信息；信息不足要直说；
5) 结尾列出引用：格式“参考：[S1][S2]…”。

# 输出格式（照抄并填充）
- 核心结论：
- 建议/步骤：
- 参考：[S1] [S2] …
"""
