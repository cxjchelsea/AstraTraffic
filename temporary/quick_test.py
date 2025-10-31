# -*- coding: utf-8 -*-
"""
quick_test.py
一键测试：意图识别 + 多知识库 RAG（交互式）
- 自动检查/补齐多库索引（扫描 data/knowledge 下的一级子目录）
- 先跑意图识别（intent/intent_adapter.py 里的 predict_intent）
- 根据 rag_retriever.INTENT_TO_KB 路由到具体库并检索
- 当无高置信命中时：明确提示“没有把握”，不乱答
"""

import os
import argparse
from typing import Dict, Tuple, List, Any, Optional

# ========== 1) 多库入库自检 ==========
def ensure_ingest_multi(root_kb_dir: str = "data/knowledge"):
    """扫描 data/knowledge/* 作为库名，检查 storage/<kb_name>/index.faiss，缺哪个就入库哪个。"""
    if not os.path.isdir(root_kb_dir):
        raise RuntimeError(f"知识库目录不存在：{root_kb_dir}")

    kb_dirs: List[Tuple[str, str]] = []
    for name in os.listdir(root_kb_dir):
        full = os.path.join(root_kb_dir, name)
        if os.path.isdir(full):
            kb_dirs.append((name, full))

    if not kb_dirs:
        raise RuntimeError(f"在 {root_kb_dir} 下未发现任何子目录，请创建如 health/、report/ 等。")

    missing = []
    for kb_name, _ in kb_dirs:
        idx_path = os.path.join("../data/storage", kb_name, "index.faiss")
        if not os.path.exists(idx_path):
            missing.append(kb_name)

    if not missing:
        print("✅ 已检测到所有知识库的索引文件，跳过入库。")
        return

    print("⚙️ 部分索引缺失，开始入库缺失的库：", ", ".join(missing))
    from modules.retriever.rag_retriever import KBIngestor, DenseEncoder
    embedder = DenseEncoder()
    ing = KBIngestor(embedder)
    for kb_name, kb_dir in kb_dirs:
        idx_path = os.path.join("../data/storage", kb_name, "index.faiss")
        if os.path.exists(idx_path):
            print(f"  - {kb_name}: 已存在，跳过")
            continue
        print(f"  - {kb_name}: 从 {kb_dir} 入库…")
        os.makedirs(os.path.join("../data/storage", kb_name), exist_ok=True)
        ing.ingest(kb_dir, kb_name=kb_name)
    print("✅ 入库完成。")


# ========== 2) 组件缓存 ==========
_SEARCHERS: Dict[str, Any] = {}

def get_cached_searcher(kb_name: str,
                        device: Optional[str] = None,
                        use_bm25: bool = True,
                        use_reranker: bool = True):
    """按库名复用检索器，避免重复加载模型与索引。"""
    if kb_name in _SEARCHERS:
        return _SEARCHERS[kb_name]
    from modules.retriever.rag_retriever import KnowledgeSearcher, DenseEncoder, _build_reranker
    embedder = DenseEncoder(device=device)
    reranker = _build_reranker(device=device) if use_reranker else None
    searcher = KnowledgeSearcher(embedder, reranker, use_bm25=use_bm25, kb_name=kb_name)
    _SEARCHERS[kb_name] = searcher
    return searcher


# ========== 3) 打印工具 ==========
def print_intent(label: str, conf: float, topk: List[Tuple[str, float]], routed: str, kb_routed: Optional[str]):
    print(f"➡️ 主意图：{label}（{conf:.3f}）")
    if topk:
        tops = " | ".join([f"{l}:{p:.3f}" for l, p in topk[:5]])
        print(f"   TopK：{tops}")
    if routed != label:
        print(f"   低置信度兜底 → 使用路由意图：{routed}")
    print("   路由到 KB：", kb_routed or "不检索")

def pretty_print_hits(hits: List[Dict[str, Any]], max_chars: int = 200):
    if not hits:
        print("🤔 我没有把握能从知识库里给出正确答案。要不要换个问法，或让我只给出一般性建议？")
        return
    for i, h in enumerate(hits, 1):
        txt = (h.get("text") or "").replace("\n", " ")
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "…"
        print(f"[{i}] score={h.get('score', 0.0):.3f} | source={h.get('source','')}")
        print(f"    {txt}")


# ========== 4) 单次查询处理 ==========
def handle_query(query: str,
                 predict_intent,
                 intent_to_kb: Dict[str, Optional[str]],
                 conf_threshold: float,
                 fallback_intent: str,
                 device: Optional[str],
                 use_bm25: bool,
                 use_reranker: bool,
                 top_k_final: int):
    # 1) 意图识别
    label, conf, topk = predict_intent(query)
    routed_intent = label if conf >= conf_threshold else fallback_intent
    kb_name = intent_to_kb.get(routed_intent, None)
    print_intent(label, conf, topk, routed_intent, kb_name)

    # 2) 路由决策
    if not kb_name:
        print("💬 当前意图不走 KB（通常交给 LLM 或其他业务流程）。")
        return

    # 3) 检索
    print("🔎 正在检索知识库...")
    searcher = get_cached_searcher(kb_name, device=device, use_bm25=use_bm25, use_reranker=use_reranker)
    hits = searcher.search(query, top_k_final=top_k_final)
    pretty_print_hits(hits)


# ========== 5) 主程序 ==========
def main():
    parser = argparse.ArgumentParser("quick_test：意图识别 + 多知识库 RAG（交互式）")
    parser.add_argument("--kb_root", type=str, default="data/knowledge", help="知识库根目录（其下每个子目录为一个库）")
    parser.add_argument("--device", type=str, default=None, help="cuda 或 cpu（默认自动）")
    parser.add_argument("--no_bm25", action="store_true", help="关闭 BM25 融合")
    parser.add_argument("--no_reranker", action="store_true", help="关闭跨编码器重排")
    parser.add_argument("--conf_th", type=float, default=0.40, help="意图置信度阈值（低于则兜底）")
    parser.add_argument("--fallback_intent", type=str, default="闲聊其他", help="低置信度兜底意图")
    parser.add_argument("--top_k_final", type=int, default=4, help="最终展示的文段数")
    parser.add_argument("--once", type=str, default=None, help="单次查询并退出")
    args = parser.parse_args()

    # 检查/补齐各库索引
    ensure_ingest_multi(args.kb_root)

    # 懒加载组件
    from process.intent_adapter import predict_intent
    from modules.retriever.rag_retriever import INTENT_TO_KB

    # 单次模式
    if args.once:
        handle_query(
            args.once, predict_intent, INTENT_TO_KB,
            conf_threshold=args.conf_th,
            fallback_intent=args.fallback_intent,
            device=args.device,
            use_bm25=(not args.no_bm25),
            use_reranker=(not args.no_reranker),
            top_k_final=args.top_k_final
        )
        return

    # 交互模式
    print("\n================== 意图识别 + RAG 检索 · 交互模式 ==================")
    print("输入你的问题并回车；输入 exit / quit / q 退出。")
    print("=================================================================\n")

    try:
        while True:
            q = input(">>> ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit", "q"}:
                print("👋 再见！")
                break

            handle_query(
                q, predict_intent, INTENT_TO_KB,
                conf_threshold=args.conf_th,
                fallback_intent=args.fallback_intent,
                device=args.device,
                use_bm25=(not args.no_bm25),
                use_reranker=(not args.no_reranker),
                top_k_final=args.top_k_final
            )
            print()
    except (EOFError, KeyboardInterrupt):
        print("\n👋 再见！")


if __name__ == "__main__":
    main()
