# -*- coding: utf-8 -*-
"""
对话历史管理模块（内存存储）
管理多轮对话的历史记录
"""
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from collections import defaultdict
import threading


class ChatHistoryManager:
    """对话历史管理器（内存存储）"""
    
    def __init__(self, max_history_per_session: int = 10):
        """
        初始化对话历史管理器
        
        Args:
            max_history_per_session: 每个会话保留的最大历史轮数
        """
        self._histories: Dict[str, List[Tuple[str, str, datetime]]] = defaultdict(list)
        self._lock = threading.Lock()  # 线程安全
        self.max_history_per_session = max_history_per_session
    
    def add_exchange(self, session_id: str, user_query: str, assistant_answer: str) -> None:
        """
        添加一轮对话
        
        Args:
            session_id: 会话ID
            user_query: 用户问题
            assistant_answer: 助手回答
        """
        with self._lock:
            history = self._histories[session_id]
            history.append((user_query, assistant_answer, datetime.now()))
            
            # 限制历史长度
            if len(history) > self.max_history_per_session:
                history.pop(0)
    
    def get_history(self, session_id: str, max_turns: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        获取会话的历史对话
        
        Args:
            session_id: 会话ID
            max_turns: 最多返回的轮数（None表示返回全部）
        
        Returns:
            历史对话列表，格式为 [(用户问题, 助手回答), ...]，按时间从旧到新
        """
        with self._lock:
            history = self._histories.get(session_id, [])
            
            if max_turns is None:
                return [(q, a) for q, a, _ in history]
            else:
                # 返回最近的 max_turns 轮
                recent = history[-max_turns:] if len(history) > max_turns else history
                return [(q, a) for q, a, _ in recent]
    
    def clear_history(self, session_id: str) -> None:
        """
        清空指定会话的历史
        
        Args:
            session_id: 会话ID
        """
        with self._lock:
            if session_id in self._histories:
                del self._histories[session_id]
    
    def clear_all(self) -> None:
        """清空所有会话的历史"""
        with self._lock:
            self._histories.clear()
    
    def get_recent_queries(self, session_id: str, max_queries: int = 3) -> List[str]:
        """
        获取最近的用户问题列表（用于查询改写）
        
        Args:
            session_id: 会话ID
            max_queries: 最多返回的问题数
        
        Returns:
            最近的用户问题列表，按时间从旧到新
        """
        with self._lock:
            history = self._histories.get(session_id, [])
            recent = history[-max_queries:] if len(history) > max_queries else history
            return [q for q, _, _ in recent]


# 全局单例实例
_history_manager = ChatHistoryManager(max_history_per_session=10)


def get_history_manager() -> ChatHistoryManager:
    """获取全局历史管理器实例"""
    return _history_manager

