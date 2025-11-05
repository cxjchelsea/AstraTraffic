# process/settings.py
# -*- coding: utf-8 -*-
"""
全局配置中心
- 自动定位项目根（包含 data/ 且包含 process/ 或 modules/）
- 加载项目根 .env（不存在也不报错）
- 统一做路径归一化为绝对路径
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# ---------- 1) 工具：类型与路径解析 ----------
def _as_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _as_int(v: str | None, default: int) -> int:
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _as_float(v: str | None, default: float) -> float:
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

def _abs_from_root(p: str | os.PathLike | None, project_root: Path, default: Path) -> Path:
    """环境变量中的路径若为相对路径，则基于 project_root 转为绝对路径；若为空则用 default。"""
    if p is None or str(p).strip() == "":
        return default.resolve()
    pth = Path(p)
    return (pth if pth.is_absolute() else (project_root / pth)).resolve()

# ---------- 2) 自动定位项目根 ----------
def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    # 逐级向上查找：满足 data/ 存在 且 (process/ 或 modules/) 存在 即视为项目根
    for p in [here.parent, *here.parents]:
        try:
            has_data = (p / "data").exists()
            has_process = (p / "process").exists()
            has_modules = (p / "modules").exists()
            if has_data and (has_process or has_modules):
                return p
        except Exception:
            pass
    # 兜底：返回 settings.py 的父级
    return here.parent

# 允许外部强制指定（绝对或相对路径都可以）
_env_root = os.getenv("PROJECT_ROOT", "")
if _env_root:
    _project_root_candidate = Path(_env_root)
    PROJECT_ROOT: Path = (_project_root_candidate if _project_root_candidate.is_absolute()
                          else (_find_project_root() / _project_root_candidate)).resolve()
else:
    PROJECT_ROOT: Path = _find_project_root().resolve()

# ---------- 3) 读取 .env ----------
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ---------- 4) 目录结构（统一归一化） ----------
DATA_DIR: Path      = _abs_from_root(os.getenv("DATA_DIR"),      PROJECT_ROOT, PROJECT_ROOT / "data")
MODELS_DIR: Path    = _abs_from_root(os.getenv("MODELS_DIR"),    PROJECT_ROOT, DATA_DIR / "models")
STORAGE_DIR: Path   = _abs_from_root(os.getenv("STORAGE_DIR"),   PROJECT_ROOT, DATA_DIR / "storage")
KNOWLEDGE_DIR: Path = _abs_from_root(os.getenv("KNOWLEDGE_DIR"), PROJECT_ROOT, DATA_DIR / "knowledge")

# 必要目录幂等创建（不会影响已有目录）
for p in (DATA_DIR, MODELS_DIR, STORAGE_DIR, KNOWLEDGE_DIR):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# ---------- 5) 检索模型 ----------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL:  str = os.getenv("RERANKER_MODEL",  "BAAI/bge-reranker-base")

# ---------- 6) 意图分类器 ----------
INTENT_BASE_MODEL_PATH: str = os.getenv("INTENT_BASE_MODEL_PATH", str((MODELS_DIR / "bert-base-chinese").resolve()))
INTENT_ADAPTER_DIR:     str = os.getenv("INTENT_ADAPTER_DIR",     str((MODELS_DIR / "ds_single").resolve()))
LABEL_MAP_PATH:         str = os.getenv("LABEL_MAP_PATH",         str((PROJECT_ROOT / "label_map_intent.json").resolve()))

# ---------- 7) LLM 配置 ----------
LLM_MODE:       str   = os.getenv("LLM_MODE", "ollama").lower()   # openai | ollama | http
RAG_LLM_MODEL:  str   = os.getenv("RAG_LLM_MODEL", "qwen2.5:14b")
OPENAI_API_KEY: str   = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL:str   = os.getenv("OPENAI_BASE_URL", "")
LLM_API_BASE_URL: str = os.getenv("LLM_API_BASE_URL", "http://localhost:8080")
LLM_HTTP_TIMEOUT:float= _as_float(os.getenv("LLM_HTTP_TIMEOUT"), 120.0)

# ---------- 8) 检索行为参数 ----------
USE_BM25:     bool  = _as_bool(os.getenv("USE_BM25"), True)
USE_RERANKER: bool  = _as_bool(os.getenv("USE_RERANKER"), True)
CONF_TH:      float = _as_float(os.getenv("CONF_TH"), 0.40)
TOP_K_FINAL:  int   = _as_int(os.getenv("TOP_K_FINAL"), 4)

# ---------- 9) 是否使用规则意图（临时跑链路） ----------
USE_RULE_INTENT: bool = _as_bool(os.getenv("USE_RULE_INTENT"), False)

# ---------- 10) Tool选择器配置 ----------
USE_TOOL_SELECTOR: bool = _as_bool(os.getenv("USE_TOOL_SELECTOR"), True)  # 默认使用tool选择器（仅LLM模式）

# ---------- 11) 实时路况API配置（高德地图） ----------
AMAP_API_KEY: str = os.getenv("AMAP_API_KEY", "")
AMAP_DEFAULT_CITY: str = os.getenv("AMAP_DEFAULT_CITY", "北京市")  # 默认城市名称
AMAP_DEFAULT_SHOW_TRAFFIC: bool = _as_bool(os.getenv("AMAP_DEFAULT_SHOW_TRAFFIC"), True)  # 默认是否显示实时路况图层
