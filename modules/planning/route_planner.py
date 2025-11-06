# -*- coding: utf-8 -*-
"""
路径规划模块 - 高德地图API
"""
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import requests

from modules.planning.route_types import RouteType, RouteStrategy


@dataclass
class RouteStep:
    """路径步骤信息"""
    instruction: str  # 行驶指示
    road: str  # 道路名称
    distance: float  # 距离（米）
    duration: float  # 耗时（秒）
    polyline: str  # 坐标串（用于绘制路径）


@dataclass
class RouteInfo:
    """路径信息数据类"""
    origin: Dict[str, float]  # 起点坐标 {lng, lat}
    destination: Dict[str, float]  # 终点坐标 {lng, lat}
    distance: float  # 总距离（米）
    duration: float  # 总耗时（秒）
    strategy: str  # 路径策略
    tolls: float  # 过路费（元）
    toll_distance: float  # 收费路段距离（米）
    steps: List[RouteStep]  # 路径步骤列表
    polyline: str  # 完整路径坐标串


class AmapRoutePlanner:
    """高德地图路径规划API"""
    
    BASE_URL = "https://restapi.amap.com/v3"
    
    def __init__(self, api_key: str):
        """
        初始化高德地图路径规划API客户端
        
        Args:
            api_key: 高德地图API密钥
        """
        self.api_key = api_key
    
    def plan_route(
        self,
        origin: str,
        destination: str,
        route_type: RouteType = "driving",
        strategy: RouteStrategy = "fastest",
        city: Optional[str] = None,
        waypoints: Optional[List[str]] = None,
    ) -> Optional[RouteInfo]:
        """
        规划路径
        
        Args:
            origin: 起点（地址或坐标，格式：经度,纬度）
            destination: 终点（地址或坐标，格式：经度,纬度）
            route_type: 路径类型（driving/walking/transit/riding）
            strategy: 路径策略（fastest/shortest/avoid_traffic等）
            city: 城市名称（可选，用于地址解析）
            waypoints: 途经点列表（可选）
        
        Returns:
            RouteInfo对象，如果查询失败返回None
        """
        try:
            # 根据路径类型选择不同的API端点
            if route_type == "transit":
                return self._plan_transit_route(origin, destination, city)
            elif route_type == "walking":
                return self._plan_walking_route(origin, destination, city)
            elif route_type == "riding":
                return self._plan_riding_route(origin, destination, city)
            else:  # driving
                return self._plan_driving_route(origin, destination, strategy, city, waypoints)
        except Exception as e:
            print(f"路径规划时发生异常: {e}")
            return None
    
    def _plan_driving_route(
        self,
        origin: str,
        destination: str,
        strategy: RouteStrategy = "fastest",
        city: Optional[str] = None,
        waypoints: Optional[List[str]] = None,
    ) -> Optional[RouteInfo]:
        """规划驾车路径"""
        try:
            # 先解析起点和终点坐标（如果传入的是地址名称，需要先进行地理编码）
            origin_coords = self._parse_coordinates(origin)
            dest_coords = self._parse_coordinates(destination)
            
            if not origin_coords:
                # 如果origin是地址名称，先进行地理编码
                origin_coords = self._geocode_address(origin, city)
                if not origin_coords:
                    print(f"无法解析起点坐标: {origin}")
                    return None
            
            if not dest_coords:
                # 如果destination是地址名称，先进行地理编码
                dest_coords = self._geocode_address(destination, city)
                if not dest_coords:
                    print(f"无法解析终点坐标: {destination}")
                    return None
            
            # 将坐标转换为API需要的格式（经度,纬度）
            origin_str = f"{origin_coords[0]},{origin_coords[1]}"
            dest_str = f"{dest_coords[0]},{dest_coords[1]}"
            
            url = f"{self.BASE_URL}/direction/driving"
            params = {
                "key": self.api_key,
                "origin": origin_str,
                "destination": dest_str,
                "strategy": self._get_strategy_code(strategy),
                "extensions": "all",  # 返回详细信息
                "output": "json",
            }
            
            if city:
                params["city"] = city
            
            if waypoints:
                # 如果途经点也是地址，也需要转换为坐标
                waypoint_coords = []
                for waypoint in waypoints:
                    wp_coords = self._parse_coordinates(waypoint) or self._geocode_address(waypoint, city)
                    if wp_coords:
                        waypoint_coords.append(f"{wp_coords[0]},{wp_coords[1]}")
                if waypoint_coords:
                    params["waypoints"] = "|".join(waypoint_coords)
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("status") != "1":
                error_info = data.get('info', '未知错误')
                print(f"高德API路径规划失败: {error_info}")
                print(f"  请求参数: origin={origin_str}, destination={dest_str}, city={city}")
                return None
            
            route_data = data.get("route", {})
            paths = route_data.get("paths", [])
            
            if not paths:
                print("未找到路径方案")
                return None
            
            # 选择第一个路径方案（最优方案）
            path = paths[0]
            
            # 解析路径步骤
            steps = []
            steps_data = path.get("steps", [])
            all_polyline_points = []  # 用于合并所有步骤的polyline
            
            for step_data in steps_data:
                step_polyline = step_data.get("polyline", "")
                step = RouteStep(
                    instruction=step_data.get("instruction", ""),
                    road=step_data.get("road", ""),
                    distance=float(step_data.get("distance", 0)),
                    duration=float(step_data.get("duration", 0)),
                    polyline=step_polyline,
                )
                steps.append(step)
                
                # 合并步骤的polyline（如果存在）
                if step_polyline:
                    # 如果all_polyline_points不为空，需要跳过第一个点（避免重复）
                    step_points = step_polyline.split(';')
                    if all_polyline_points:
                        all_polyline_points.extend(step_points[1:])  # 跳过第一个点
                    else:
                        all_polyline_points.extend(step_points)
            
            # 构建完整的polyline（如果path中没有，则使用合并后的步骤polyline）
            full_polyline = path.get("polyline", "")
            if not full_polyline and all_polyline_points:
                full_polyline = ';'.join(all_polyline_points)
            
            # 验证路径：检查polyline是否有效
            if full_polyline:
                polyline_points = full_polyline.split(';')
                print(f"[INFO] 路径验证: 总点数={len(polyline_points)}, 起点坐标={origin_coords}, 终点坐标={dest_coords}")
                
                # 验证起点和终点是否在路径附近（允许一定误差）
                if polyline_points:
                    first_point = polyline_points[0].split(',')
                    last_point = polyline_points[-1].split(',')
                    if len(first_point) == 2 and len(last_point) == 2:
                        first_lng, first_lat = float(first_point[0]), float(first_point[1])
                        last_lng, last_lat = float(last_point[0]), float(last_point[1])
                        
                        # 计算起点和终点与路径首尾点的距离（粗略估算，单位：度）
                        origin_diff = abs(first_lng - origin_coords[0]) + abs(first_lat - origin_coords[1])
                        dest_diff = abs(last_lng - dest_coords[0]) + abs(last_lat - dest_coords[1])
                        
                        # 如果距离超过0.01度（约1公里），给出警告
                        if origin_diff > 0.01:
                            print(f"[WARNING] 路径起点与查询起点距离较远: {origin_diff:.6f}度")
                        if dest_diff > 0.01:
                            print(f"[WARNING] 路径终点与查询终点距离较远: {dest_diff:.6f}度")
            
            return RouteInfo(
                origin={"lng": origin_coords[0], "lat": origin_coords[1]},
                destination={"lng": dest_coords[0], "lat": dest_coords[1]},
                distance=float(path.get("distance", 0)),
                duration=float(path.get("duration", 0)),
                strategy=strategy,
                tolls=float(path.get("tolls", 0)),
                toll_distance=float(path.get("toll_distance", 0)),
                steps=steps,
                polyline=full_polyline,
            )
        except Exception as e:
            print(f"规划驾车路径时发生异常: {e}")
            return None
    
    def _plan_walking_route(
        self,
        origin: str,
        destination: str,
        city: Optional[str] = None,
    ) -> Optional[RouteInfo]:
        """规划步行路径"""
        try:
            # 先解析起点和终点坐标
            origin_coords = self._parse_coordinates(origin) or self._geocode_address(origin, city)
            dest_coords = self._parse_coordinates(destination) or self._geocode_address(destination, city)
            
            if not origin_coords:
                print(f"无法解析起点坐标: {origin}")
                return None
            if not dest_coords:
                print(f"无法解析终点坐标: {destination}")
                return None
            
            # 将坐标转换为API需要的格式
            origin_str = f"{origin_coords[0]},{origin_coords[1]}"
            dest_str = f"{dest_coords[0]},{dest_coords[1]}"
            
            url = f"{self.BASE_URL}/direction/walking"
            params = {
                "key": self.api_key,
                "origin": origin_str,
                "destination": dest_str,
                "extensions": "all",
                "output": "json",
            }
            
            if city:
                params["city"] = city
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("status") != "1":
                error_info = data.get('info', '未知错误')
                print(f"高德API步行路径规划失败: {error_info}")
                print(f"  请求参数: origin={origin_str}, destination={dest_str}, city={city}")
                return None
            
            route_data = data.get("route", {})
            paths = route_data.get("paths", [])
            
            if not paths:
                return None
            
            path = paths[0]
            
            steps = []
            steps_data = path.get("steps", [])
            all_polyline_points = []  # 用于合并所有步骤的polyline
            
            for step_data in steps_data:
                step_polyline = step_data.get("polyline", "")
                step = RouteStep(
                    instruction=step_data.get("instruction", ""),
                    road=step_data.get("road", ""),
                    distance=float(step_data.get("distance", 0)),
                    duration=float(step_data.get("duration", 0)),
                    polyline=step_polyline,
                )
                steps.append(step)
                
                # 合并步骤的polyline（如果存在）
                if step_polyline:
                    # 如果all_polyline_points不为空，需要跳过第一个点（避免重复）
                    step_points = step_polyline.split(';')
                    if all_polyline_points:
                        all_polyline_points.extend(step_points[1:])  # 跳过第一个点
                    else:
                        all_polyline_points.extend(step_points)
            
            # 构建完整的polyline（如果path中没有，则使用合并后的步骤polyline）
            full_polyline = path.get("polyline", "")
            if not full_polyline and all_polyline_points:
                full_polyline = ';'.join(all_polyline_points)
            
            return RouteInfo(
                origin={"lng": origin_coords[0], "lat": origin_coords[1]},
                destination={"lng": dest_coords[0], "lat": dest_coords[1]},
                distance=float(path.get("distance", 0)),
                duration=float(path.get("duration", 0)),
                strategy="shortest",
                tolls=0.0,
                toll_distance=0.0,
                steps=steps,
                polyline=full_polyline,
            )
        except Exception as e:
            print(f"规划步行路径时发生异常: {e}")
            return None
    
    def _plan_riding_route(
        self,
        origin: str,
        destination: str,
        city: Optional[str] = None,
    ) -> Optional[RouteInfo]:
        """规划骑行路径"""
        try:
            # 先解析起点和终点坐标
            origin_coords = self._parse_coordinates(origin) or self._geocode_address(origin, city)
            dest_coords = self._parse_coordinates(destination) or self._geocode_address(destination, city)
            
            if not origin_coords:
                print(f"无法解析起点坐标: {origin}")
                return None
            if not dest_coords:
                print(f"无法解析终点坐标: {destination}")
                return None
            
            # 将坐标转换为API需要的格式
            origin_str = f"{origin_coords[0]},{origin_coords[1]}"
            dest_str = f"{dest_coords[0]},{dest_coords[1]}"
            
            url = f"{self.BASE_URL}/direction/bicycling"
            params = {
                "key": self.api_key,
                "origin": origin_str,
                "destination": dest_str,
                "extensions": "all",
                "output": "json",
            }
            
            if city:
                params["city"] = city
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("status") != "1":
                error_info = data.get('info', '未知错误')
                print(f"高德API骑行路径规划失败: {error_info}")
                print(f"  请求参数: origin={origin_str}, destination={dest_str}, city={city}")
                return None
            
            route_data = data.get("route", {})
            paths = route_data.get("paths", [])
            
            if not paths:
                return None
            
            path = paths[0]
            
            steps = []
            steps_data = path.get("steps", [])
            all_polyline_points = []  # 用于合并所有步骤的polyline
            
            for step_data in steps_data:
                step_polyline = step_data.get("polyline", "")
                step = RouteStep(
                    instruction=step_data.get("instruction", ""),
                    road=step_data.get("road", ""),
                    distance=float(step_data.get("distance", 0)),
                    duration=float(step_data.get("duration", 0)),
                    polyline=step_polyline,
                )
                steps.append(step)
                
                # 合并步骤的polyline（如果存在）
                if step_polyline:
                    # 如果all_polyline_points不为空，需要跳过第一个点（避免重复）
                    step_points = step_polyline.split(';')
                    if all_polyline_points:
                        all_polyline_points.extend(step_points[1:])  # 跳过第一个点
                    else:
                        all_polyline_points.extend(step_points)
            
            # 构建完整的polyline（如果path中没有，则使用合并后的步骤polyline）
            full_polyline = path.get("polyline", "")
            if not full_polyline and all_polyline_points:
                full_polyline = ';'.join(all_polyline_points)
            
            return RouteInfo(
                origin={"lng": origin_coords[0], "lat": origin_coords[1]},
                destination={"lng": dest_coords[0], "lat": dest_coords[1]},
                distance=float(path.get("distance", 0)),
                duration=float(path.get("duration", 0)),
                strategy="shortest",
                tolls=0.0,
                toll_distance=0.0,
                steps=steps,
                polyline=full_polyline,
            )
        except Exception as e:
            print(f"规划骑行路径时发生异常: {e}")
            return None
    
    def _plan_transit_route(
        self,
        origin: str,
        destination: str,
        city: Optional[str] = None,
    ) -> Optional[RouteInfo]:
        """规划公交路径（简化实现，返回基础信息）"""
        # 公交路径规划比较复杂，这里先返回None，后续可以扩展
        print("公交路径规划功能暂未实现")
        return None
    
    def _parse_coordinates(self, location: str) -> Optional[tuple]:
        """
        解析坐标字符串（格式：经度,纬度）
        
        Args:
            location: 坐标字符串或地址
        
        Returns:
            (经度, 纬度) 元组，如果不是坐标格式返回None
        """
        try:
            parts = location.split(",")
            if len(parts) == 2:
                lng = float(parts[0].strip())
                lat = float(parts[1].strip())
                return (lng, lat)
        except (ValueError, AttributeError):
            pass
        return None
    
    def _geocode_address(self, address: str, city: Optional[str] = None) -> Optional[tuple]:
        """
        地理编码：将地址转换为坐标
        
        Args:
            address: 地址字符串
            city: 城市名称（可选）
        
        Returns:
            (经度, 纬度) 元组，如果失败返回None
        """
        try:
            url = f"{self.BASE_URL}/geocode/geo"
            params = {
                "key": self.api_key,
                "address": address,
                "output": "json",
            }
            
            if city:
                params["city"] = city
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("status") != "1":
                return None
            
            geocodes = data.get("geocodes", [])
            if not geocodes:
                return None
            
            location_str = geocodes[0].get("location", "")
            return self._parse_coordinates(location_str)
        except Exception as e:
            print(f"地理编码时发生异常: {e}")
            return None
    
    def _get_strategy_code(self, strategy: RouteStrategy) -> str:
        """
        获取路径策略代码（高德地图API使用）
        
        Args:
            strategy: 路径策略
        
        Returns:
            策略代码字符串
        """
        strategy_map = {
            "fastest": "0",  # 速度优先（时间最短）
            "shortest": "2",  # 距离最短
            "avoid_traffic": "3",  # 不走高速
            "avoid_highway": "4",  # 避免拥堵
            "avoid_toll": "5",  # 不走高速且避免拥堵
        }
        return strategy_map.get(strategy, "0")


# 全局客户端实例（单例模式）
_route_planner_instance: Optional[AmapRoutePlanner] = None


def get_route_planner(api_key: Optional[str] = None) -> Optional[AmapRoutePlanner]:
    """
    获取路径规划API客户端实例（单例）
    
    Args:
        api_key: API密钥（可选，如果不提供则从配置读取）
    
    Returns:
        AmapRoutePlanner实例，如果未配置API密钥则返回None
    """
    global _route_planner_instance
    
    if _route_planner_instance is not None:
        return _route_planner_instance
    
    from modules.config.settings import AMAP_API_KEY
    
    key = api_key or AMAP_API_KEY
    if not key:
        return None
    
    _route_planner_instance = AmapRoutePlanner(key)
    return _route_planner_instance

