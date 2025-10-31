# modules/retriever/ingest_kb.py
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

from modules.retriever.rag_retriever import KBIngestor, DenseEncoder, storage_paths
from settings import DATA_DIR  # <<< 关键：用全局 DATA_DIR

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
    paths = storage_paths(kb_name)
    return Path(paths["index"]).exists() and Path(paths["meta"]).exists()

def main():
    parser = argparse.ArgumentParser("批量入库知识库")
    default_root = (DATA_DIR / "knowledge").resolve()  # <<< 默认根目录
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
