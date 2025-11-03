# -*- coding: utf-8 -*-
"""
LangChain 版本的 CLI
使用 LangChain 架构的交互式命令行工具
"""
from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv()  # 会自动读取项目根目录的 .env

# 直接使用 LangChain 实现
from lc.rag_chain import rag_answer_langchain as rag_answer


def main():
    print("\n================== RAG + LLM (LangChain) · 交互模式 ==================")
    print("使用 LangChain 架构")
    print("输入问题回车；exit/quit/q 退出。")
    print("=========================================================\n")
    try:
        while True:
            q = input(">>> ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit", "q"}:
                print("👋 再见！")
                break
            
            pack = rag_answer(q)
            print(f"➡️ 意图：{pack.intent.label}（{pack.intent.score:.3f}） · 路由KB：{pack.kb or '不检索'}")
            if not pack.hits:
                print("🤔 没有把握，建议换个问法或补充信息。\n")
                continue
            
            print("\n====== 答案 ======\n" + pack.answer)
            print("\n====== 证据 ======")
            for i, h in enumerate(pack.hits, 1):
                t = h.text.replace("\n", " ")
                if len(t) > 160:
                    t = t[:160] + "…"
                print(f"[S{i}] {t}  ({h.source})")
            print()
    except (KeyboardInterrupt, EOFError):
        print("\n👋 再见！")


if __name__ == "__main__":
    main()

