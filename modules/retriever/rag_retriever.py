#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 检索模块（中文友好，支持多知识库 + 混合检索 + 重排 + 置信门槛）
- 入库：本地 FAISS 向量库 + （可选）BM25 词匹配
- 检索：稠密向量 + 稀疏 BM25 融合 + （可选）跨编码器重排
- 置信门槛：无把握时返回空（不乱答）

依赖（推荐）：
  pip install -U sentence-transformers faiss-cpu rank-bm25 jieba tqdm
  # 可选（重排器，推荐）
  pip install -U transformers torch

目录结构（多库）：
  data/knowledge/
    health/
    report/
  storage/
    health/index.faiss, meta.jsonl, bm25_tokens.json
    report/index.faiss, meta.jsonl, bm25_tokens.json

CLI：
  入库（指定库）：
    python rag_retriever.py --ingest data/knowledge/health --kb_name health
  检索（指定库）：
    python rag_retriever.py --query "高血压的一线用药有哪些？" --kb_name health
"""

import os
import re
import json
import csv
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Optional, Tuple

import jieba

# ---- 可选依赖的“柔性导入” ----
try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # noqa

try:
    import torch  # type: ignore
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # noqa
    torch = None  # noqa

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None  # noqa

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
except Exception:
    AutoTokenizer = None  # noqa
    AutoModelForSequenceClassification = None  # noqa


# ================= 可调配置 =================
from settings import (
    STORAGE_DIR, EMBEDDING_MODEL, RERANKER_MODEL
)


MAX_CHARS_PER_CHUNK = 500                    # 分块目标长度
CHUNK_OVERLAP       = 100                    # 分块重叠
TOP_K               = 6                      # 初检召回
TOP_K_FINAL         = 4                      # 最终返回条数
HYBRID_WEIGHT       = 0.35                   # 稠密/稀疏融合权重（越大越偏 BM25）

# ---- 置信门槛（无把握就返回空） ----
MIN_DENSE_SIM    = 0.30   # 稠密相似度（归一化内积/余弦）下限
MIN_HYBRID_SCORE = 0.25   # 融合分数（0~1）下限
RERANKER_PROB_TH = 0.55   # 重排概率阈值（sigmoid 后）
REQUIRE_AT_LEAST = 1      # 至少需要多少条高置信命中

# ---- 多库路径助手 ----
from pathlib import Path

def storage_paths(kb_name: str):
    base = Path(STORAGE_DIR) / kb_name
    return {
        "base": str(base),
        "index": str(base / "index.faiss"),
        "meta": str(base / "meta.jsonl"),
        "bm25": str(base / "bm25_tokens.json"),
    }


# ---- 意图 → 库名路由（与你实际目录保持一致：health / report） ----
INTENT_TO_KB: Dict[str, Optional[str]] = {
    "健康咨询": "health",
    "报告解读": "report",
    "药品服务": None,
    "环境健康": None,
    "就医转诊": None,
    "紧急求助": None,
    "情感支持": None,
    "闲聊其他": None,
}


# ================= 数据结构 =================
@dataclass
class DocChunk:
    doc_id: str
    chunk_id: int
    text: str
    source: str
    meta: Dict[str, Any]


# ================= 文本分块 =================
SPLIT_PAT = re.compile(r"(?<=[。！？；;.!?\n])")

def split_zh(text: str, target_len: int = MAX_CHARS_PER_CHUNK, overlap: int = CHUNK_OVERLAP) -> List[str]:
    sentences = [s.strip() for s in SPLIT_PAT.split(text) if s and s.strip()]
    chunks: List[str] = []
    buff: List[str] = []
    cur_len = 0
    for s in sentences:
        if cur_len + len(s) <= target_len:
            buff.append(s); cur_len += len(s)
        else:
            if buff:
                chunks.append("".join(buff))
                tail = chunks[-1][-overlap:] if overlap > 0 else ""
                buff = ([tail, s] if tail else [s])
                cur_len = len(tail) + len(s)
            else:
                # 极长句硬切
                for i in range(0, len(s), target_len):
                    chunks.append(s[i:i+target_len])
                buff = []; cur_len = 0
    if buff:
        chunks.append("".join(buff))
    return chunks


# ================= 语料加载 =================
def load_text_from_path(path: str) -> str:
    if path.endswith((".txt", ".md")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if path.endswith(".jsonl"):
        texts = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    txt = obj.get("text") or obj.get("content") or json.dumps(obj, ensure_ascii=False)
                    texts.append(str(txt))
                except Exception:
                    continue
        return "\n".join(texts)
    if path.endswith(".csv"):
        texts = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                txt = row.get("text") or row.get("content") or " ".join(row.values())
                texts.append(str(txt))
        return "\n".join(texts)
    return ""


def iter_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".txt", ".md", ".jsonl", ".csv")):
                yield os.path.join(dirpath, fn)


# ================= 表示层 =================
class DenseEncoder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: Optional[str] = None):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers 未安装：pip install -U sentence-transformers")
        # trust_remote_code=True 能安静加载，并兼容非原生 sbert 的 HF 模型目录
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        if device:
            self.model = self.model.to(device)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


class SparseBM25:
    def __init__(self, corpus_tokens: List[List[str]]):
        if BM25Okapi is None:
            raise ImportError("rank-bm25 未安装：pip install rank-bm25")
        self.bm25 = BM25Okapi(corpus_tokens)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        return list(self.bm25.get_scores(query_tokens))


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL, device: Optional[str] = None):
        if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
            raise ImportError("缺少 transformers/torch：pip install -U transformers torch")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if device:
            self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, query: str, candidates: List[str], batch_size: int = 8) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            enc = self.tok([query]*len(batch), batch, truncation=True, padding=True, return_tensors='pt')
            if next(self.model.parameters()).is_cuda:
                enc = {k: v.cuda() for k, v in enc.items()}
            logits = self.model(**enc).logits.view(-1)
            probs = torch.sigmoid(logits)   # ← 转成 [0,1] 概率
            scores.extend(probs.detach().cpu().tolist())
        return scores


# ================= 向量索引 =================
class FaissStore:
    def __init__(self, dim: int):
        if faiss is None:
            raise ImportError("faiss 未安装：pip install faiss-cpu")
        self.index = faiss.IndexFlatIP(dim)

    def add(self, embeddings: List[List[float]]):
        import numpy as np
        vecs = np.array(embeddings, dtype='float32')
        self.index.add(vecs)

    def search(self, query_emb: List[float], top_k: int) -> Tuple[List[int], List[float]]:
        import numpy as np
        q = np.array(query_emb, dtype='float32')[None, :]
        sims, idx = self.index.search(q, top_k)
        return idx[0].tolist(), sims[0].tolist()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str) -> "FaissStore":
        if faiss is None:
            raise ImportError("faiss 未安装：pip install faiss-cpu")
        index = faiss.read_index(path)
        store = cls(index.d)
        store.index = index
        return store


# ================= 入库流程 =================
class KBIngestor:
    def __init__(self, embedder: DenseEncoder):
        self.embedder = embedder

    def ingest(self, input_dir: str, kb_name: str = "default") -> None:
        paths = storage_paths(kb_name)
        os.makedirs(paths["base"], exist_ok=True)

        chunks: List[DocChunk] = []
        print(f"[Ingest] scanning: {input_dir}")
        for path in iter_files(input_dir):
            raw = load_text_from_path(path)
            if not raw.strip():
                continue
            doc_id = os.path.relpath(path, start=input_dir)
            for i, ch in enumerate(split_zh(raw)):
                meta = {"doc_id": doc_id, "chunk_id": i, "source": path, "kb_name": kb_name}
                chunks.append(DocChunk(doc_id, i, ch, path, meta))

        if not chunks:
            print("[Ingest] no chunks found.")
            return

        texts = [c.text for c in chunks]
        print(f"[Ingest] encoding {len(texts)} chunks with {EMBEDDING_MODEL} ...")
        embs = self.embedder.encode(texts)

        print("[Ingest] building FAISS index ...")
        store = FaissStore(dim=len(embs[0]))
        store.add(embs)
        store.save(paths["index"])

        print(f"[Ingest] writing meta: {paths['meta']}")
        with open(paths["meta"], "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({"text": c.text, **c.meta}, ensure_ascii=False) + "\n")

        if BM25Okapi is not None:
            print("[Ingest] building BM25 corpus ...")
            corpus_tokens = [list(jieba.cut(t)) for t in texts]
            with open(paths["bm25"], "w", encoding="utf-8") as f:
                json.dump(corpus_tokens, f, ensure_ascii=False)

        print(f"[Ingest] done. kb={kb_name}")


# ================= 检索流程 =================
class KnowledgeSearcher:
    def __init__(self,
                 embedder: Optional[DenseEncoder] = None,
                 reranker: Optional[CrossEncoderReranker] = None,
                 use_bm25: bool = True,
                 hybrid_weight: float = HYBRID_WEIGHT,
                 kb_name: str = "default"):
        self.embedder = embedder
        self.reranker = reranker
        self.use_bm25 = use_bm25 and (BM25Okapi is not None)
        self.hybrid_weight = hybrid_weight
        self.kb_name = kb_name

        paths = storage_paths(kb_name)
        self.store = FaissStore.load(paths["index"])
        self.metas: List[Dict[str, Any]] = []
        with open(paths["meta"], "r", encoding="utf-8") as f:
            for line in f:
                self.metas.append(json.loads(line))

        if self.use_bm25:
            if os.path.exists(paths["bm25"]):
                with open(paths["bm25"], "r", encoding="utf-8") as f:
                    corpus_tokens = json.load(f)
            else:
                corpus_tokens = [list(jieba.cut(m["text"])) for m in self.metas]
            self.bm25 = SparseBM25(corpus_tokens)
        else:
            self.bm25 = None

    def _hybrid_rank(self, dense_scores: List[float], sparse_scores: Optional[List[float]]) -> List[Tuple[int, float]]:
        idxs = list(range(len(dense_scores)))
        if sparse_scores is None:
            return sorted([(i, dense_scores[i]) for i in idxs], key=lambda x: x[1], reverse=True)

        # Min-Max 归一化到 [0,1]
        def minmax(arr: List[float]) -> List[float]:
            if not arr:
                return []
            mn, mx = min(arr), max(arr)
            if math.isclose(mn, mx):
                return [0.0 for _ in arr]
            return [(x - mn) / (mx - mn) for x in arr]

        d = minmax(dense_scores)
        s = minmax(sparse_scores)
        fused = [self.hybrid_weight * s[i] + (1 - self.hybrid_weight) * d[i] for i in idxs]
        return sorted([(i, fused[i]) for i in idxs], key=lambda x: x[1], reverse=True)

    def search(self, query: str, top_k: int = TOP_K, top_k_final: int = TOP_K_FINAL) -> List[Dict[str, Any]]:
        if self.embedder is None:
            raise RuntimeError("search 需要 embedder，请在初始化时传入 DenseEncoder")

        # 1) 稠密检索
        q_emb = self.embedder.encode([query])[0]
        idxs, sims = self.store.search(q_emb, top_k=max(top_k, 20))

        # 早停：最大稠密相似度过低 → 视为不可答
        if len(sims) == 0 or max(sims) < MIN_DENSE_SIM:
            return []

        # 2) 稀疏 BM25（可选）
        sparse_scores = None
        if self.use_bm25 and self.bm25 is not None:
            q_tokens = list(jieba.cut(query))
            sparse_scores = self.bm25.get_scores(q_tokens)

        # 3) 融合打分（在稠密召回集合上融合）
        dense_subset = [sims[i] for i in range(len(idxs))]
        sparse_subset = [sparse_scores[idx] for idx in idxs] if sparse_scores else None
        fused_pairs = self._hybrid_rank(dense_subset, sparse_subset)

        # 过滤低融合分数
        fused_pairs = [(i, sc) for i, sc in fused_pairs if sc >= MIN_HYBRID_SCORE]
        if not fused_pairs:
            return []

        # 取前 top_k
        top_pairs = fused_pairs[:top_k]
        cand_idxs = [idxs[i] for i, _ in top_pairs]
        candidates = [self.metas[i] for i in cand_idxs]

        # 4) 跨编码器重排（若启用）
        if self.reranker is not None:
            texts = [c["text"] for c in candidates]
            probs = self.reranker.score(query, texts)  # [0,1]
            reranked = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)
            reranked = [(c, p) for c, p in reranked if p >= RERANKER_PROB_TH][:top_k_final]

            if len(reranked) < REQUIRE_AT_LEAST:
                return []

            return [
                {"text": c["text"], "score": float(p), "source": c.get("source"),
                 "doc_id": c.get("doc_id"), "chunk_id": c.get("chunk_id")}
                for c, p in reranked
            ]

        # 否则：用融合分数截断
        kept = [(cand_idxs[j], top_pairs[j][1]) for j in range(min(len(top_pairs), top_k_final))]
        if len(kept) < REQUIRE_AT_LEAST:
            return []
        return [
            {"text": self.metas[i]["text"], "score": float(sc), "source": self.metas[i].get("source"),
             "doc_id": self.metas[i].get("doc_id"), "chunk_id": self.metas[i].get("chunk_id")}
            for i, sc in kept
        ]


# =================（可选）路由封装 =================
def route_and_retrieve(intent_label: str, query: str, get_searcher, top_k_final: int = TOP_K_FINAL) -> List[Dict[str, Any]]:
    kb_name = INTENT_TO_KB.get(intent_label, None)
    if not kb_name:
        return []
    searcher: KnowledgeSearcher = get_searcher(kb_name)
    return searcher.search(query, top_k_final=top_k_final)


# ================= CLI =================
def _build_embedder(device: Optional[str] = None) -> DenseEncoder:
    return DenseEncoder(EMBEDDING_MODEL, device=device)

def _build_reranker(device: Optional[str] = None) -> Optional[CrossEncoderReranker]:
    try:
        return CrossEncoderReranker(RERANKER_MODEL, device=device)
    except Exception:
        print("[Warn] 未启用重排（缺少 transformers/torch）。建议安装以提升效果。")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser("RAG 检索模块（多库 + 置信门槛）")
    parser.add_argument("--ingest",  type=str, default=None, help="入库目录（包含 .txt/.md/.jsonl/.csv）")
    parser.add_argument("--query",   type=str, default=None, help="检索问题")
    parser.add_argument("--kb_name", type=str, default="default", help="知识库名称（storage/<kb_name>/）")
    parser.add_argument("--device",  type=str, default=None, help="cuda 或 cpu（默认自动）")
    parser.add_argument("--no_bm25", action="store_true", help="关闭 BM25 融合")
    args = parser.parse_args()

    if args.ingest:
        embedder = _build_embedder(device=args.device)
        KBIngestor(embedder).ingest(args.ingest, kb_name=args.kb_name)

    if args.query:
        embedder = _build_embedder(device=args.device)
        reranker = _build_reranker(device=args.device)
        searcher = KnowledgeSearcher(embedder, reranker, use_bm25=(not args.no_bm25), kb_name=args.kb_name)
        results = searcher.search(args.query)
        print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
