# -*- coding: utf-8 -*-
"""
FastAPI 主应用
提供 RESTful API 接口
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from core.rag_chain import rag_answer_with_history

# 创建 FastAPI 应用
app = FastAPI(
    title="AstraTraffic API",
    description="智能出行管家系统 API",
    version="1.0.0"
)

# 配置 CORS（允许前端跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 请求/响应模型 ====================

class ChatRequest(BaseModel):
    """聊天请求模型"""
    query: str
    session_id: str = "default"


class HitResponse(BaseModel):
    """检索结果响应模型"""
    text: str
    score: float
    source: str
    doc_id: Optional[str] = None
    chunk_id: Optional[int] = None


class MapDataResponse(BaseModel):
    """地图数据响应模型"""
    location: Dict[str, float]  # {lng: float, lat: float}
    zoom: Optional[int] = 15
    markers: Optional[List[Dict[str, Any]]] = None
    show_traffic: Optional[bool] = False
    location_name: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: str
    intent: str
    intent_score: float
    kb: Optional[str] = None
    hits: List[HitResponse] = []
    map_data: Optional[MapDataResponse] = None
    metrics: Dict[str, Any] = {}


# ==================== API 路由 ====================

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "name": "AstraTraffic API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """健康检查接口"""
    return {"status": "healthy"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口
    
    接收用户查询，返回智能回答和相关数据（包括地图数据）
    """
    try:
        # 调用现有的 RAG 链
        pack = rag_answer_with_history(
            query=request.query,
            session_id=request.session_id
        )
        
        # 提取地图数据（如果存在）
        map_data = None
        if pack.metrics.notes and "map_data" in pack.metrics.notes:
            map_data_raw = pack.metrics.notes["map_data"]
            if map_data_raw:
                try:
                    map_data = MapDataResponse(**map_data_raw)
                    print(f"[API] 成功提取地图数据: {map_data_raw}")
                except Exception as e:
                    print(f"[API ERROR] 地图数据格式错误: {e}, 原始数据: {map_data_raw}")
        else:
            print(f"[API] metrics.notes中没有map_data, notes内容: {pack.metrics.notes}")
        
        # 构建响应
        response = ChatResponse(
            answer=pack.answer,
            intent=pack.intent.label,
            intent_score=pack.intent.score,
            kb=pack.kb,
            hits=[
                HitResponse(
                    text=h.text,
                    score=h.score,
                    source=h.source,
                    doc_id=h.doc_id,
                    chunk_id=h.chunk_id
                )
                for h in pack.hits
            ],
            map_data=map_data,
            metrics={
                "used_kb": pack.metrics.used_kb,
                "intent_conf": pack.metrics.intent_conf,
                "conf_th": pack.metrics.conf_th,
                "notes": {k: v for k, v in (pack.metrics.notes or {}).items() 
                         if k != "map_data"}  # 排除map_data，已单独提取
            }
        )
        
        return response
        
    except Exception as e:
        import traceback
        error_detail = str(e)
        error_traceback = traceback.format_exc()
        
        # 打印详细错误信息到控制台（用于调试）
        print(f"[ERROR] API请求处理失败:")
        print(error_traceback)
        
        # 返回用户友好的错误信息
        raise HTTPException(
            status_code=500,
            detail=f"处理请求时发生错误: {error_detail}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

