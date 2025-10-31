from typing import List
from process.types import Hit

def build_prompt(query: str, hits: List[Hit]) -> str:
    """统一的提示词模板；可按库/意图做分支增强。"""
    ctx_lines = []
    for i,h in enumerate(hits, 1):
        text = (h.text or "").replace("\n"," ").strip()
        ctx_lines.append(f"[S{i}] {text} (source: {h.source})")
    ctx_block = "\n".join(ctx_lines)

    return f"""你是严谨的中文医学助理。仅依据“资料”回答；如果资料不足，请明确说“根据现有资料无法给出确定答案”。

# 问题
{query}

# 资料（可能不完整）
{ctx_block}

# 要求
1) 先用3–5句给出核心结论，语言简洁；
2) 若涉及用药或报告判读，必须提示“需结合个体情况并在医生指导下进行”；
3) 不得杜撰资料外的信息；
4) 结尾列出引用：参考：[S1][S2]…
"""
