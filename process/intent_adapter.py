# intent_adapter.py
# 将你的命令行脚本里的 PrimaryInfer 包装成项目内可复用接口
from typing import Tuple, List
import threading

# ⬇⬇⬇ 改成你“命令行脚本”的模块名（即包含 PrimaryInfer 的那个 .py 文件名）
# 例如你的文件名是 cli_infer.py，就写：from cli_infer import PrimaryInfer
from modules.intent.intent_classify import PrimaryInfer   # <<< 修改点

# 单例 + 线程安全懒加载，避免重复占用显存/内存
_loader_lock = threading.Lock()
_singleton: PrimaryInfer | None = None

def _get_model() -> PrimaryInfer:
    global _singleton
    if _singleton is None:
        with _loader_lock:
            if _singleton is None:
                _singleton = PrimaryInfer()  # 使用你脚本中的默认路径与配置
    return _singleton

def predict_intent(text: str) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    返回：(主意图标签, 置信度, TopK 列表[(label, prob)...])
    """
    clf = _get_model()
    out = clf.predict_one(text)
    label, prob = out["primary"]
    topk = out["topk_primary"]
    return label, float(prob), [(l, float(p)) for l, p in topk]
