# -*- coding: utf-8 -*-
"""
RAG 系统的数据类型定义（通用数据结构，不依赖特定框架）
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class IntentTop:
    label: str
    score: float

@dataclass
class IntentResult:
    label: str
    score: float
    topk: List[IntentTop]

@dataclass
class Hit:
    text: str
    score: float
    source: str
    doc_id: Optional[str] = None
    chunk_id: Optional[int] = None

@dataclass
class Metrics:
    used_kb: Optional[str] = None
    intent_conf: float = 0.0
    conf_th: float = 0.0
    notes: Dict[str, Any] = None

@dataclass
class AnswerPack:
    intent: IntentResult
    kb: Optional[str]
    hits: List[Hit]
    answer: str
    metrics: Metrics


