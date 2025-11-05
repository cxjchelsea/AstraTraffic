# -*- coding: utf-8 -*-
"""
输入处理业务逻辑
职责：接收和管理用户输入、对话历史等
"""
from typing import List, Tuple
from modules.chat.history import ChatHistoryManager, get_history_manager as _get_history_manager


def get_history_manager() -> ChatHistoryManager:
    """获取全局历史管理器实例"""
    return _get_history_manager()


# 导出ChatHistoryManager以保持向后兼容
from modules.chat.history import ChatHistoryManager


