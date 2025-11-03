# -*- coding: utf-8 -*-
"""
LangChain Retriever 适配器
直接调用 modules.retriever，不通过 process adapter
"""
from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from modules.retriever.rag_retriever import (
    KnowledgeSearcher, DenseEncoder, _build_reranker, INTENT_TO_KB
)
from rag_types import Hit
from settings import USE_BM25, USE_RERANKER, TOP_K_FINAL

# 轻量缓存，避免重复加载索引/模型
_SEARCHERS: Dict[str, KnowledgeSearcher] = {}


def get_searcher(
    kb_name: str,
    device: Optional[str] = None,
    use_bm25: bool = USE_BM25,
    use_reranker: bool = USE_RERANKER,
) -> KnowledgeSearcher:
    """按 KB 名获取/构建单例检索器（直接调用 modules）"""
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
    """统一的检索接口：返回 Hit 列表（直接调用 modules）"""
    searcher = get_searcher(kb_name, device=device, use_bm25=use_bm25, use_reranker=use_reranker)
    rows = searcher.search(query, top_k_final=top_k_final)

    # 将底层返回（dict）映射为 Hit
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


class CustomRetriever(BaseRetriever):
    """将现有检索系统适配为 LangChain Retriever（直接调用 modules）"""
    
    kb_name: str = Field(description="知识库名称")
    device: Optional[str] = Field(default=None, description="设备类型")
    use_bm25: bool = Field(default=USE_BM25, description="是否使用 BM25")
    use_reranker: bool = Field(default=USE_RERANKER, description="是否使用重排器")
    top_k_final: int = Field(default=TOP_K_FINAL, description="最终返回文档数")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """检索相关文档"""
        hits = retrieve(
            query=query,
            kb_name=self.kb_name,
            top_k_final=self.top_k_final,
            device=self.device,
            use_bm25=self.use_bm25,
            use_reranker=self.use_reranker
        )
        
        # 转换为 LangChain Document 格式
        documents = []
        for hit in hits:
            doc = Document(
                page_content=hit.text,
                metadata={
                    "source": hit.source or "",
                    "score": hit.score,
                    "doc_id": hit.doc_id,
                    "chunk_id": hit.chunk_id,
                }
            )
            documents.append(doc)
        
        return documents
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步检索相关文档"""
        return self._get_relevant_documents(query)


class IntentAwareRetriever(BaseRetriever):
    """基于意图识别的智能检索器（直接调用 modules）"""
    
    device: Optional[str] = Field(default=None, description="设备类型")
    use_bm25: bool = Field(default=USE_BM25, description="是否使用 BM25")
    use_reranker: bool = Field(default=USE_RERANKER, description="是否使用重排器")
    top_k_final: int = Field(default=TOP_K_FINAL, description="最终返回文档数")
    conf_th: float = Field(default=0.40, description="意图置信度阈值")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _route_to_kb(self, intent_label: str, query: str, conf: float) -> Optional[str]:
        """根据意图路由到知识库"""
        # 高置信度：使用意图映射
        if conf >= self.conf_th:
            return INTENT_TO_KB.get(intent_label)
        
        # 低置信度：关键词兜底路由
        _FALLBACK_KEYWORDS = [
            (("限行", "处罚", "罚款", "记分", "专用道", "电子警察"), "law"),
            (("信号", "配时", "相位", "绿信比", "潮汐车道", "可变车道", "诱导"), "handbook"),
            (("公交", "地铁", "轨道", "换乘", "票价", "首班", "末班", "到站"), "transit"),
            (("停车", "泊位", "车场", "路侧"), "parking"),
            (("充电", "快充", "直流", "交流", "电桩"), "ev"),
            (("车路协同", "RSU", "OBU", "C-V2X", "V2X"), "iov"),
        ]
        
        q = (query or "").lower()
        for kws, kb in _FALLBACK_KEYWORDS:
            for k in kws:
                if k.lower() in q:
                    return kb
        
        return INTENT_TO_KB.get("闲聊其他")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """基于意图识别的检索"""
        # 使用统一的意图识别适配器
        from lc.intent_adapter import predict_intent
        intent_label, conf, _ = predict_intent(query)
        
        # 路由到知识库
        kb_name = self._route_to_kb(intent_label, query, conf)
        
        if not kb_name:
            return []
        
        # 检索
        hits = retrieve(
            query=query,
            kb_name=kb_name,
            top_k_final=self.top_k_final,
            device=self.device,
            use_bm25=self.use_bm25,
            use_reranker=self.use_reranker
        )
        
        # 转换为 LangChain Document 格式
        documents = []
        for hit in hits:
            doc = Document(
                page_content=hit.text,
                metadata={
                    "source": hit.source or "",
                    "score": hit.score,
                    "doc_id": hit.doc_id,
                    "chunk_id": hit.chunk_id,
                    "kb_name": kb_name,
                    "intent": intent_label,
                    "intent_conf": conf,
                }
            )
            documents.append(doc)
        
        return documents
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步检索相关文档"""
        return self._get_relevant_documents(query)
