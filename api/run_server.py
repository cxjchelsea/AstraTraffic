#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动 FastAPI 服务器
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式，自动重载
        log_level="info"
    )

