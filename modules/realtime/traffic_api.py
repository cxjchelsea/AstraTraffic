# -*- coding: utf-8 -*-
"""
实时路况数据接入模块 - 高德地图API
"""
from typing import Optional
from dataclasses import dataclass
import requests
import time


@dataclass
class TrafficInfo:
    """路况信息数据类"""
    road_name: str  # 道路名称
    city: str  # 城市
    congestion_level: str  # 拥堵等级：畅通、缓行、拥堵、严重拥堵
    speed: Optional[float] = None  # 平均速度（km/h）
    status_code: Optional[int] = None  # 状态码：0-畅通，1-缓行，2-拥堵，3-严重拥堵
    description: Optional[str] = None  # 详细描述
    timestamp: Optional[float] = None  # 时间戳


class AmapTrafficAPI:
    """高德地图实时路况API"""
    
    BASE_URL = "https://restapi.amap.com/v3"
    
    def __init__(self, api_key: str):
        """
        初始化高德地图API客户端
        
        Args:
            api_key: 高德地图API密钥
        """
        self.api_key = api_key
    
    def get_traffic_by_road(self, road_name: str, city: str = "北京市") -> Optional[TrafficInfo]:
        """
        根据道路名称查询实时路况
        
        Args:
            road_name: 道路名称，例如"中关村大街"
            city: 城市名称，默认为"北京市"
            
        Returns:
            TrafficInfo对象，如果查询失败返回None
        """
        try:
            # 第一步：搜索道路POI，获取坐标
            search_url = f"{self.BASE_URL}/place/text"
            params = {
                "key": self.api_key,
                "keywords": road_name,
                "city": city,
                "types": "190301",  # 道路类型编码
                "output": "json"
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            data = response.json()
            
            # 检查是否成功
            if data.get("status") != "1":
                print(f"高德API搜索失败: {data.get('info', '未知错误')}")
                return None
            
            pois = data.get("pois", [])
            if not pois:
                print(f"未找到道路: {road_name} 在 {city}")
                return None
            
            # 获取第一个匹配的道路坐标
            poi = pois[0]
            location_str = poi.get("location", "")
            if not location_str:
                return None
            
            # 解析坐标 "经度,纬度"
            location = location_str.split(",")
            if len(location) != 2:
                return None
            
            longitude = location[0]
            latitude = location[1]
            
            # 第二步：使用坐标查询实时路况
            # 高德地图的交通态势API需要矩形区域，我们用一个很小的矩形（即一个点）
            traffic_url = f"{self.BASE_URL}/traffic/status/rectangle"
            params = {
                "key": self.api_key,
                "rectangle": f"{longitude},{latitude};{longitude},{latitude}",
                "level": 6,  # 道路级别
                "extensions": "all"  # 返回详细信息
            }
            
            response = requests.get(traffic_url, params=params, timeout=10)
            traffic_data = response.json()
            
            if traffic_data.get("status") != "1":
                print(f"高德API路况查询失败: {traffic_data.get('info', '未知错误')}")
                return None
            
            # 解析路况数据
            traffic_info = traffic_data.get("trafficinfo", {})
            roads = traffic_info.get("roads", [])
            
            if not roads:
                print(f"未找到道路的实时路况数据")
                return None
            
            # 取第一条道路的路况信息
            road = roads[0]
            status = road.get("status", "0")  # 状态码：0-畅通，1-缓行，2-拥堵，3-严重拥堵
            speed = road.get("speed", "0")
            
            # 状态码转拥堵等级
            status_map = {
                "0": "畅通",
                "1": "缓行",
                "2": "拥堵",
                "3": "严重拥堵"
            }
            
            congestion_level = status_map.get(status, "未知")
            
            # 构建返回结果
            return TrafficInfo(
                road_name=road_name,
                city=city,
                congestion_level=congestion_level,
                speed=float(speed) if speed and speed != "0" else None,
                status_code=int(status) if status.isdigit() else None,
                description=f"平均速度: {speed}km/h" if speed and speed != "0" else "速度信息暂无",
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"查询路况时发生异常: {e}")
            return None

