# -*- coding: utf-8 -*-
"""
路径规划业务逻辑
职责：统一管理路径规划相关的业务逻辑
包含：
- 起点终点提取（LLM）
- 路径规划执行
- 路径数据格式化
"""
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from modules.planning import AmapRoutePlanner, RouteInfo, get_route_planner
from modules.planning.route_types import RouteType, RouteStrategy
from modules.config.settings import AMAP_API_KEY, AMAP_DEFAULT_CITY
from services.tool_selector import ToolSelection


# ==================== 数据格式化函数 ====================

def format_route_info(route: RouteInfo, route_type: str = "driving") -> str:
    """
    格式化路径信息为文本（用于RAG上下文）
    
    Args:
        route: RouteInfo对象
        route_type: 路径类型（driving/walking/riding/transit）
    
    Returns:
        格式化的文本字符串
    """
    parts = []
    
    # 路径类型说明
    route_type_names = {
        "driving": "驾车",
        "walking": "步行",
        "riding": "骑行",
        "transit": "公交",
    }
    route_type_name = route_type_names.get(route_type, "驾车")
    
    # 距离和时间（明确说明是实际路径距离，不是直线距离）
    distance_km = route.distance / 1000
    duration_min = route.duration / 60
    
    parts.append(f"出行方式：{route_type_name}")
    parts.append(f"实际路径距离：{distance_km:.2f}公里（这是实际可行驶的路径距离，不是直线距离）")
    parts.append(f"{route_type_name}预计时间：{duration_min:.1f}分钟（这是{route_type_name}的预计时间）")
    
    if route.tolls > 0:
        parts.append(f"过路费：{route.tolls:.2f}元")
    
    # 路径策略说明
    strategy_names = {
        "fastest": "最快路径",
        "shortest": "最短路径",
        "avoid_traffic": "避开拥堵",
        "avoid_highway": "不走高速",
        "avoid_toll": "不走收费",
    }
    strategy_name = strategy_names.get(route.strategy, route.strategy)
    parts.append(f"路径策略：{strategy_name}")
    
    return " | ".join(parts)


def format_route_info_to_dict(route: RouteInfo) -> Dict[str, Any]:
    """
    将RouteInfo对象转换为字典格式（用于API返回）
    
    Args:
        route: RouteInfo对象
    
    Returns:
        字典格式的路径数据
    """
    return {
        "origin": route.origin,
        "destination": route.destination,
        "distance": route.distance,
        "duration": route.duration,
        "strategy": route.strategy,
        "tolls": route.tolls,
        "toll_distance": route.toll_distance,
        "polyline": route.polyline,
        "steps": [
            {
                "instruction": step.instruction,
                "road": step.road,
                "distance": step.distance,
                "duration": step.duration,
                "polyline": step.polyline,
            }
            for step in route.steps
        ],
    }


# ==================== 信息提取函数 ====================

def extract_route_type_from_query(query: str) -> str:
    """
    从用户查询中提取出行方式（使用LLM智能提取）
    
    Args:
        query: 用户查询文本
    
    Returns:
        出行方式字符串（driving/walking/riding/transit），默认返回"driving"
    """
    query_lower = query.lower()
    
    # 规则匹配：检查查询中是否包含出行方式关键词（按优先级匹配）
    # 注意：先匹配更具体的词，避免"走"误匹配
    if any(keyword in query_lower for keyword in ["步行", "走路", "徒步"]):
        return "walking"
    elif any(keyword in query_lower for keyword in ["骑行", "骑车", "自行车", "单车"]):
        return "riding"
    elif any(keyword in query_lower for keyword in ["公交", "地铁", "公共交通", "坐车", "乘车"]):
        return "transit"
    elif any(keyword in query_lower for keyword in ["驾车", "开车", "自驾"]):
        return "driving"
    
    # 如果没有明确的关键词，使用LLM提取
    try:
        from modules.generator.prompt import build_route_type_extraction_prompt
        from adapters.llm import get_llm_client
        
        prompt = build_route_type_extraction_prompt(query)
        llm_client = get_llm_client()
        response = llm_client(prompt, temperature=0.1, max_tokens=20)
        
        # 清理响应
        route_type = response.strip().strip('"').strip("'").strip().lower()
        
        # 映射到标准格式
        route_type_map = {
            "驾车": "driving",
            "开车": "driving",
            "自驾": "driving",
            "步行": "walking",
            "走路": "walking",
            "徒步": "walking",
            "骑行": "riding",
            "骑车": "riding",
            "自行车": "riding",
            "公交": "transit",
            "地铁": "transit",
            "公共交通": "transit",
            "driving": "driving",
            "walking": "walking",
            "riding": "riding",
            "transit": "transit",
        }
        
        route_type = route_type_map.get(route_type, "driving")
        print(f"[INFO] LLM提取出行方式: {route_type}")
        return route_type
        
    except Exception as e:
        print(f"[WARNING] LLM提取出行方式失败: {e}，使用默认值: driving")
        return "driving"


def extract_origin_destination_from_query(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从用户查询中提取起点和终点（使用LLM智能提取）
    
    Args:
        query: 用户查询文本
    
    Returns:
        (起点, 终点) 元组，如果未找到返回(None, None)
    """
    try:
        from modules.generator.prompt import build_route_extraction_prompt
        from adapters.llm import get_llm_client
        
        prompt = build_route_extraction_prompt(query)
        llm_client = get_llm_client()
        response = llm_client(prompt, temperature=0.1, max_tokens=50)
        
        # 解析响应：格式应该是"起点|终点"
        result = response.strip().strip('"').strip("'").strip()
        
        if not result or result.lower() in ["无", "none", "null", ""]:
            return (None, None)
        
        # 分割起点和终点
        parts = result.split("|")
        if len(parts) == 2:
            origin = parts[0].strip()
            destination = parts[1].strip()
            
            # 检查是否表示"无"
            if origin.lower() in ["无", "none", "null", ""]:
                origin = None
            if destination.lower() in ["无", "none", "null", ""]:
                destination = None
            
            return (origin, destination)
        else:
            # 如果格式不对，尝试其他解析方式
            print(f"[WARNING] 起点终点提取格式异常: {result}")
            return (None, None)
            
    except Exception as e:
        print(f"[WARNING] LLM提取起点终点失败: {e}")
        return (None, None)


# ==================== 路径规划执行器 ====================

@dataclass
class PlanningToolResult:
    """路径规划工具执行结果"""
    success: bool
    context_text: str  # 添加到上下文的文本
    metadata: Dict[str, Any] = None  # 额外的元数据（包含路径数据）


class PlanningToolExecutor:
    """路径规划工具执行器"""
    
    def __init__(self):
        """初始化路径规划工具执行器"""
        pass
    
    def execute_planning(
        self,
        tool_selection: ToolSelection,
        query: str,
        route_type: RouteType = "driving",
        strategy: RouteStrategy = "fastest",
        city: Optional[str] = None,
        **kwargs
    ) -> Optional[PlanningToolResult]:
        """
        执行路径规划
        
        Args:
            tool_selection: Tool选择结果
            query: 用户查询
            route_type: 路径类型（driving/walking/transit/riding）
            strategy: 路径策略（fastest/shortest/avoid_traffic等）
            city: 城市名称（可选，默认使用配置中的城市）
            **kwargs: 额外参数
        
        Returns:
            PlanningToolResult对象，如果执行失败返回None
        """
        default_city = city or AMAP_DEFAULT_CITY
        
        # 从查询中提取起点和终点
        origin, destination = extract_origin_destination_from_query(query)
        
        if not origin or not destination:
            context_text = (
                "【路径规划】\n"
                "无法从查询中提取起点和终点。请提供明确的起点和终点，例如：\n"
                "- '从北京站到天安门怎么走'\n"
                "- '帮我规划从中关村到西单的路线'\n"
                "- '从天安门到故宫'"
            )
            return PlanningToolResult(
                success=False,
                context_text=context_text,
                metadata={"reason": "no_origin_destination"}
            )
        
        # 如果没有指定route_type，从查询中提取
        if route_type == "driving" and "route_type" not in kwargs:
            extracted_type = extract_route_type_from_query(query)
            if extracted_type != "driving":
                route_type = extracted_type
                print(f"[INFO] 从查询中提取出行方式: {route_type}")
        
        print(f"[INFO] 提取到起点: {origin}, 终点: {destination}, 出行方式: {route_type}, 城市: {default_city}")
        
        # 获取路径规划客户端
        route_planner = get_route_planner()
        if not route_planner:
            context_text = "【路径规划】\n路径规划服务暂时不可用，请检查AMAP_API_KEY配置。"
            return PlanningToolResult(
                success=False,
                context_text=context_text,
                metadata={"reason": "api_not_available"}
            )
        
        # 调用路径规划API
        route_info = route_planner.plan_route(
            origin=origin,
            destination=destination,
            route_type=route_type,
            strategy=strategy,
            city=default_city,
        )
        
        if route_info:
            print(f"[INFO] 成功规划路径: {route_info.distance/1000:.2f}公里, {route_info.duration/60:.1f}分钟")
            
            route_text = format_route_info(route_info, route_type=route_type)
            context_text = (
                f"【路径规划】\n"
                f"起点：{origin}\n"
                f"终点：{destination}\n"
                f"{route_text}"
            )
            
            route_data_dict = format_route_info_to_dict(route_info)
            
            return PlanningToolResult(
                success=True,
                context_text=context_text,
                metadata={
                    "origin": origin,
                    "destination": destination,
                    "city": default_city,
                    "route_type": route_type,
                    "strategy": strategy,
                    "route_data": route_data_dict,
                }
            )
        else:
            print(f"[WARNING] 未能规划路径: {origin} 到 {destination}")
            context_text = (
                f"【路径规划】\n"
                f"未能规划从'{origin}'到'{destination}'的路径。可能原因：\n"
                f"1) 起点或终点名称不准确（请尝试使用完整地名）\n"
                f"2) API服务暂时不可用（请检查AMAP_API_KEY配置）\n"
                f"3) 该路径暂无法规划"
            )
            return PlanningToolResult(
                success=False,
                context_text=context_text,
                metadata={
                    "origin": origin,
                    "destination": destination,
                    "city": default_city,
                    "reason": "api_failed"
                }
            )


# ==================== 全局单例和便捷函数 ====================

_planning_executor_instance: Optional[PlanningToolExecutor] = None


def get_planning_executor() -> PlanningToolExecutor:
    """获取全局路径规划工具执行器实例"""
    global _planning_executor_instance
    if _planning_executor_instance is None:
        _planning_executor_instance = PlanningToolExecutor()
    return _planning_executor_instance


def execute_planning_tool(
    tool_selection: ToolSelection,
    query: str,
    **kwargs
) -> Optional[PlanningToolResult]:
    """
    便捷函数：执行路径规划工具
    
    Args:
        tool_selection: Tool选择结果
        query: 用户查询
        **kwargs: 额外参数
    
    Returns:
        PlanningToolResult对象
    """
    executor = get_planning_executor()
    return executor.execute_planning(tool_selection, query, **kwargs)

