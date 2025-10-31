# process/retriever_adapter.py
# -*- coding: utf-8 -*-
"""
检索适配器：把底层实现封装成“可替换”的统一接口。
未来若改为 Java/HTTP 检索，只需改本文件。
"""
from typing import Optional, List, Dict
from process.types import Hit
# 如果你已经把实现文件移动/改名为 modules/retriever/impl_faiss.py，请同步这里的 import
from modules.retriever.rag_retriever import (
    KnowledgeSearcher, DenseEncoder, _build_reranker,
)
from modules.retriever.rag_retriever import INTENT_TO_KB as INTENT2KB_DEFAULT
from settings import USE_BM25, USE_RERANKER, TOP_K_FINAL

# 轻量缓存，避免重复加载索引/模型
_SEARCHERS: Dict[str, KnowledgeSearcher] = {}

def get_searcher(
    kb_name: str,
    device: Optional[str] = None,
    use_bm25: bool = USE_BM25,
    use_reranker: bool = USE_RERANKER,
) -> KnowledgeSearcher:
    """按 KB 名获取/构建单例检索器。"""
    if kb_name in _SEARCHERS:
        return _SEARCHERS[kb_name]
    embedder = DenseEncoder(device=device)
    reranker = _build_reranker(device=device) if use_reranker else None
    searcher = KnowledgeSearcher(embedder, reranker, use_bm25=use_bm25, kb_name=kb_name)
    _SEARCHERS[kb_name] = searcher
    return searcher

def retrieve(
    query: str,
    kb_name: str,
    top_k_final: int = TOP_K_FINAL,
    device: Optional[str] = None,
    use_bm25: bool = USE_BM25,
    use_reranker: bool = USE_RERANKER,
) -> List[Hit]:
    """
    统一的检索接口：返回 Hit 列表。
    若未来切 HTTP/Java 检索，只需在这里改成 HTTP 调用，并把返回映射到 Hit。
    """
    searcher = get_searcher(kb_name, device=device, use_bm25=use_bm25, use_reranker=use_reranker)
    rows = searcher.search(query, top_k_final=top_k_final)

    # 将底层返回（dict）映射为上层标准数据结构 Hit
    hits: List[Hit] = [
        Hit(
            text=r["text"],
            score=float(r["score"]),
            source=r.get("source", "") or "",
            doc_id=r.get("doc_id"),
            chunk_id=r.get("chunk_id"),
        )
        for r in rows
    ]
    return hits

# 暴露默认的意图=>知识库映射（上层可覆盖）
INTENT_TO_KB = dict(INTENT2KB_DEFAULT)
