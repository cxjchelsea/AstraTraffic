# -*- coding: utf-8 -*-
"""
知识库入库脚本（CLI 工具）
用于批量入库知识库文件，生成 FAISS 向量索引和 BM25 索引

用法:
    python scripts/ingest_kb.py --root data/knowledge --include law,parking
    python scripts/ingest_kb.py --root data/knowledge --rebuild
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.retriever.rag_retriever import KBIngestor, DenseEncoder, storage_paths
from modules.config.settings import DATA_DIR


def discover_kbs(root: Path):
    """发现 root 下的一级子目录作为 kb 名称（只要目录里有 .txt/.md/.jsonl/.csv 就算有效）。"""
    for p in sorted(root.iterdir()):
        if p.is_dir():
            has_file = any(
                str(fp).lower().endswith((".txt", ".md", ".jsonl", ".csv"))
                for fp in p.rglob("*")
            )
            if has_file:
                yield p.name, p


def index_exists(kb_name: str) -> bool:
    """检查索引是否已存在"""
    paths = storage_paths(kb_name)
    return Path(paths["index"]).exists() and Path(paths["meta"]).exists()


def main():
    parser = argparse.ArgumentParser(
        description="批量入库知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 入库所有知识库
  python scripts/ingest_kb.py --root data/knowledge

  # 只入库指定的知识库
  python scripts/ingest_kb.py --root data/knowledge --include law,parking

  # 强制重建索引
  python scripts/ingest_kb.py --root data/knowledge --rebuild
        """
    )
    default_root = (DATA_DIR / "knowledge").resolve() if DATA_DIR else Path("data/knowledge").resolve()
    parser.add_argument("--root", type=str, default=str(default_root), help="知识根目录")
    parser.add_argument("--include", type=str, default="", help="只入这些库（逗号分隔）")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引（否则存在则跳过）")
    parser.add_argument("--device", type=str, default=None, help="句向量设备：cuda 或 cpu")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"[INFO] KB root = {root}")
    if not root.exists():
        print(f"[ERR] root not found: {root}")
        return

    include = set([x.strip() for x in args.include.split(",") if x.strip()]) if args.include else None

    embedder = DenseEncoder(device=args.device)
    ing = KBIngestor(embedder)

    found = list(discover_kbs(root))
    if include:
        found = [kv for kv in found if kv[0] in include]

    if not found:
        print(f"[WARN] no KB found under: {root}")
        return

    for kb_name, kb_dir in found:
        print(f"\n=== Ingest KB: {kb_name} ===")
        if (not args.rebuild) and index_exists(kb_name):
            print(f"✅ detected index for {kb_name}, skip (use --rebuild to force).")
            continue
        ing.ingest(str(kb_dir), kb_name=kb_name)
    print("\n✅ All KBs processed.")


if __name__ == "__main__":
    main()


