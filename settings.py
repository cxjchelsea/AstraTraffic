# process/settings.py
# -*- coding: utf-8 -*-
"""
全局配置中心：
- 自动定位项目根（包含 data/ 与 process/ 的目录）
- 加载项目根 .env（不存在也不报错）
- 提供默认值、类型转换与路径拼装
- 所有模块统一 import 本文件的变量
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ---------- 1) 自动定位项目根 ----------
def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    # 逐级向上最多找 5 层，找到同时包含 data/ 与 process/ 的目录就认为是项目根
    for p in [here.parent, here.parent.parent, here.parent.parent.parent, here.parent.parent.parent.parent, here.parents[4] if len(here.parents) > 4 else here.parent]:
        try:
            if (p / "data").exists() and (p / "process").exists():
                return p
        except Exception:
            pass
    # 兜底：返回 process 的父级
    return here.parent

# 允许外部强制指定
_ENV_PROJECT_ROOT = os.getenv("PROJECT_ROOT", "")
PROJECT_ROOT: Path = Path(_ENV_PROJECT_ROOT) if _ENV_PROJECT_ROOT else _find_project_root()

# ---------- 2) 读取 .env ----------
# 显式加载项目根目录的 .env（若不存在，不报错）
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ---------- 3) 类型转换工具 ----------
def _as_bool(v: str, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _as_int(v: str, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default

def _as_float(v: str, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default

# ---------- 4) 目录结构（可被 .env 覆盖） ----------
DATA_DIR: Path      = Path(os.getenv("DATA_DIR",      PROJECT_ROOT / "data"))
MODELS_DIR: Path    = Path(os.getenv("MODELS_DIR",    DATA_DIR / "models"))
STORAGE_DIR: Path   = Path(os.getenv("STORAGE_DIR",   DATA_DIR / "storage"))
KNOWLEDGE_DIR: Path = Path(os.getenv("KNOWLEDGE_DIR", DATA_DIR / "knowledge"))

# 必要目录幂等创建
for p in [DATA_DIR, MODELS_DIR, STORAGE_DIR, KNOWLEDGE_DIR]:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# ---------- 5) 检索模型 ----------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL:  str = os.getenv("RERANKER_MODEL",  "BAAI/bge-reranker-base")

# ---------- 6) 意图分类器 ----------
INTENT_BASE_MODEL_PATH: str = os.getenv("INTENT_BASE_MODEL_PATH", str(MODELS_DIR / "bert-base-chinese"))
INTENT_ADAPTER_DIR:     str = os.getenv("INTENT_ADAPTER_DIR",     str(MODELS_DIR / "ds_single"))
LABEL_MAP_PATH:         str = os.getenv("LABEL_MAP_PATH",         str(PROJECT_ROOT / "label_map_intent.json"))

# ---------- 7) LLM 配置 ----------
# 模式：openai | ollama | http
LLM_MODE:       str = os.getenv("LLM_MODE", "ollama").lower()
RAG_LLM_MODEL:  str = os.getenv("RAG_LLM_MODEL", "qwen2.5:14b")

# OpenAI 兼容
OPENAI_API_KEY:  str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")

# 本地/HTTP
LLM_API_BASE_URL: str   = os.getenv("LLM_API_BASE_URL", "http://localhost:8080")
LLM_HTTP_TIMEOUT: float = _as_float(os.getenv("LLM_HTTP_TIMEOUT", "120"), 120.0)

# ---------- 8) 检索行为参数 ----------
USE_BM25:     bool  = _as_bool(os.getenv("USE_BM25", "1"), True)
USE_RERANKER: bool  = _as_bool(os.getenv("USE_RERANKER", "1"), True)
CONF_TH:      float = _as_float(os.getenv("CONF_TH", "0.40"), 0.40)
TOP_K_FINAL:  int   = _as_int(os.getenv("TOP_K_FINAL", "4"), 4)
