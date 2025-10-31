"""
LLM 适配器：OpenAI 兼容 / Ollama / 通用 HTTP。
返回一个统一的 callable: generate(prompt:str) -> str
通过环境变量切换：
  LLM_MODE=openai|ollama|http
  RAG_LLM_MODEL=模型名（如 gpt-4o-mini / qwen2.5:14b）
  OPENAI_API_KEY / OPENAI_BASE_URL（可选）
  LLM_API_BASE_URL=http://localhost:11434  （ollama）或你的HTTP服务地址
"""
import os, requests
from settings import (
    LLM_MODE, RAG_LLM_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL,
    LLM_API_BASE_URL, LLM_HTTP_TIMEOUT
)

def _mode():
    m = os.getenv("LLM_MODE", "").lower()
    if m: return m
    # 简单推断：有 OPENAI_API_KEY 就默认 openai，否则优先 ollama
    return "openai" if os.getenv("OPENAI_API_KEY") else "ollama"

def get_llm_client():
    mode = LLM_MODE
    model = RAG_LLM_MODEL

    if mode == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)

        def gen(prompt: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        return gen

    if mode == "ollama":
        base = LLM_API_BASE_URL.rstrip("/")
        timeout = LLM_HTTP_TIMEOUT
        def gen(prompt: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
            payload = {"model": model, "prompt": prompt, "options": {"temperature": temperature, "num_predict": max_tokens}, "stream": False}
            r = requests.post(f"{base}/api/generate", json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or data.get("text") or data.get("output") or "").strip()
        return gen

    # 通用 HTTP：POST 到 LLM_API_BASE_URL（你自定义后端）
    base = os.getenv("LLM_API_BASE_URL")
    if not base:
        raise RuntimeError("LLM_MODE=http 时必须设置 LLM_API_BASE_URL")
    base = LLM_API_BASE_URL.rstrip("/")
    timeout = LLM_HTTP_TIMEOUT
    def gen(prompt: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
        payload = {"prompt": prompt, "model": model, "temperature": temperature, "max_tokens": max_tokens}
        r = requests.post(base, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return (data.get("text") or data.get("output") or data.get("data") or "").strip()
    return gen
