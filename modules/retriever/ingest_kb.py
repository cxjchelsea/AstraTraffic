# ingest_kb.py
from modules.retriever.rag_retriever import KBIngestor, DenseEncoder

KB_LIST = [
    ("health", "data/knowledge/health"),
    ("report", "data/knowledge/report"),
]

if __name__ == "__main__":
    embedder = DenseEncoder()
    ing = KBIngestor(embedder)
    for kb_name, kb_dir in KB_LIST:
        print(f"\n=== Ingest KB: {kb_name} ===")
        ingest_dir = kb_dir
        ing.ingest(ingest_dir, kb_name=kb_name)
    print("\n✅ All KBs ingested.")

