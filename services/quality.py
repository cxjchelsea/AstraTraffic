# -*- coding: utf-8 -*-
"""
质量检查业务逻辑
职责：判断检索结果的质量，决定是否使用
"""
from typing import List
from langchain_core.documents import Document


def check_hits_quality(
    documents: List[Document],
    min_top_score: float = 0.15,
    min_margin: float = 0.03
) -> bool:
    """
    判断检索结果质量是否合格（决策层）
    
    Args:
        documents: 检索到的文档列表
        min_top_score: 最高分的最小阈值
        min_margin: 最高分与次高分的最小差值（可选，当前未严格使用）
    
    Returns:
        True表示质量合格，False表示不合格
    """
    if not documents:
        return False
    
    scores = [float(doc.metadata.get("score", 0.0)) for doc in documents]
    top = scores[0]
    
    if top < min_top_score:
        return False
    
    if len(scores) >= 2 and (top - scores[1]) < min_margin:
        pass  # 可以放宽一点
    
    return True

