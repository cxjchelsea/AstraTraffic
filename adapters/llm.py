# -*- coding: utf-8 -*-
"""
LLM适配器（LangChain接口适配）
职责：将底层LLM客户端适配为 LangChain BaseLanguageModel 接口
"""
from typing import Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field

import os
import requests
from modules.config.settings import (
    LLM_MODE, RAG_LLM_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL,
    LLM_API_BASE_URL, LLM_HTTP_TIMEOUT
)


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
            try:
                r = requests.post(f"{base}/api/generate", json=payload, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                return (data.get("response") or data.get("text") or data.get("output") or "").strip()
            except requests.exceptions.ConnectionError as e:
                error_msg = f"无法连接到Ollama服务 ({base})。请检查服务是否正在运行。"
                raise ConnectionError(error_msg) from e
            except requests.exceptions.Timeout as e:
                error_msg = f"Ollama服务请求超时 (超过{timeout}秒)。请检查服务状态或增加超时时间。"
                raise TimeoutError(error_msg) from e
            except requests.exceptions.HTTPError as e:
                status_code = getattr(e.response, 'status_code', '未知') if hasattr(e, 'response') and e.response is not None else "未知"
                error_detail = ""
                try:
                    if hasattr(e, 'response') and e.response is not None:
                        error_text = e.response.text[:200] if e.response.text else ""
                        if error_text:
                            error_detail = f" 错误详情: {error_text}"
                except:
                    pass
                error_msg = f"Ollama服务返回HTTP错误 {status_code}{error_detail}。请检查模型名称({model})是否正确，或服务是否正常运行。"
                raise RuntimeError(error_msg) from e
            except Exception as e:
                error_msg = f"调用Ollama服务时发生未知错误: {str(e)}"
                raise RuntimeError(error_msg) from e
        return gen

    # 通用 HTTP：POST 到 LLM_API_BASE_URL
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


class CustomLLM(LLM):
    """将现有 LLM 客户端适配为 LangChain LLM（直接实现，不通过 adapter）"""
    
    model_name: str = Field(default=RAG_LLM_MODEL)
    mode: str = Field(default=LLM_MODE)
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=700)
    
    @property
    def _llm_type(self) -> str:
        return "custom_llm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        llm_client = get_llm_client()
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        try:
            response = llm_client(prompt, temperature=temperature, max_tokens=max_tokens)
            return response
        except Exception as e:
            return f"[LLM Error: {str(e)}]"


def get_langchain_llm() -> BaseLanguageModel:
    """获取 LangChain LLM 实例（优先使用 LangChain 原生，否则使用自定义）"""
    try:
        from langchain_ollama import OllamaLLM
        from langchain_openai import ChatOpenAI
    except ImportError:
        try:
            from langchain_community.llms import Ollama as OllamaLLM
        except ImportError:
            OllamaLLM = None
    
    if LLM_MODE == "openai":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=RAG_LLM_MODEL,
                api_key=OPENAI_API_KEY or None,
                base_url=OPENAI_BASE_URL or None,
                temperature=0.2,
                max_tokens=700,
            )
        except ImportError:
            pass
    elif LLM_MODE == "ollama" and OllamaLLM is not None:
        try:
            return OllamaLLM(
                model=RAG_LLM_MODEL,
                base_url=LLM_API_BASE_URL.rstrip("/") if LLM_API_BASE_URL else "http://localhost:11434",
                temperature=0.2,
                num_predict=700,
            )
        except Exception:
            pass
    
    # 使用自定义适配器（兜底方案，直接实现）
    return CustomLLM(
        model_name=RAG_LLM_MODEL,
        mode=LLM_MODE,
        temperature=0.2,
        max_tokens=700,
    )

