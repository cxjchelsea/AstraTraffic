# -*- coding: utf-8 -*-
"""
检索适配器（LangChain接口适配）
职责：将底层检索系统适配为 LangChain BaseRetriever 接口
"""
from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from modules.retriever.rag_retriever import (
    KnowledgeSearcher, DenseEncoder, _build_reranker
)
from modules.rag_types.rag_types import Hit
from modules.config.settings import USE_BM25, USE_RERANKER, TOP_K_FINAL

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


# IntentAwareRetriever已移除，仅使用ToolAwareRetriever


class ToolAwareRetriever(BaseRetriever):
    """
    基于Tool选择器的智能检索器（执行层）
    使用LLM提示词选择tool，不依赖意图识别模型
    """
    
    device: Optional[str] = Field(default=None, description="设备类型")
    use_bm25: bool = Field(default=USE_BM25, description="是否使用 BM25")
    use_reranker: bool = Field(default=USE_RERANKER, description="是否使用重排器")
    top_k_final: int = Field(default=TOP_K_FINAL, description="最终返回文档数")
    use_llm: bool = Field(default=True, description="是否使用LLM提示词选择tool（已废弃，始终使用LLM）")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        基于Tool选择器的检索（执行层）
        使用LLM提示词选择tool，不依赖意图识别模型
        """
        # 使用决策层进行tool选择（仅LLM模式）
        from services.tool_selector import select_tool
        
        tool_selection = select_tool(query)  # 始终使用LLM
        
        # 如果是实时路况工具，不返回文档（在format_docs阶段处理）
        if tool_selection.tool == "realtime_traffic":
            return []
        
        # 如果是知识库工具，执行检索
        if tool_selection.tool.startswith("kb_") and tool_selection.kb_name:
            hits = retrieve(
                query=query,
                kb_name=tool_selection.kb_name,
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
                        "kb_name": tool_selection.kb_name,
                        "tool": tool_selection.tool,
                        "tool_reasoning": tool_selection.reasoning,
                        "tool_confidence": tool_selection.confidence,
                    }
                )
                documents.append(doc)
            
            return documents
        
        # 其他情况返回空
        return []
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步检索相关文档"""
        return self._get_relevant_documents(query)
