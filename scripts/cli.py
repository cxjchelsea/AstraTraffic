# -*- coding: utf-8 -*-
"""
LangChain 版本的 CLI
使用 LangChain 架构的交互式命令行工具
"""
from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv()  # 会自动读取项目根目录的 .env

# 使用多轮对话版本
from lc.rag_chain import rag_answer_with_history
from lc.chat_adapter import get_history_manager


def main():
    print("\n================== RAG + LLM (LangChain) · 多轮对话模式 ==================")
    print("使用 LangChain 架构，支持多轮对话和上下文理解")
    print("输入问题回车；exit/quit/q 退出；/clear 清空对话历史")
    print("=========================================================\n")
    
    session_id = "cli_session"  # 固定会话ID
    history_manager = get_history_manager()
    
    try:
        while True:
            q = input(">>> ").strip()
            if not q:
                continue
            
            # 特殊命令处理
            if q.lower() in {"exit", "quit", "q"}:
                print("👋 再见！")
                break
            
            if q.lower() in {"clear", "/clear", "/清空"}:
                history_manager.clear_history(session_id)
                print("✅ 对话历史已清空\n")
                continue
            
            # 调用多轮对话接口
            pack = rag_answer_with_history(q, session_id=session_id)
            
            # 显示改写后的查询（如果有改写）
            rewritten = pack.metrics.notes.get("rewritten_query")
            if rewritten and rewritten != q:
                print(f"📝 查询改写：{q} → {rewritten}")
            
            print(f"➡️ 意图：{pack.intent.label}（{pack.intent.score:.3f}） · 路由KB：{pack.kb or '不检索'}")
            
            # 显示历史轮数
            has_history = pack.metrics.notes.get("has_history", False)
            if has_history:
                history = history_manager.get_history(session_id, max_turns=5)
                print(f"💬 当前使用 {len(history)} 轮历史对话")
            
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

