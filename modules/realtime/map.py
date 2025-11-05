# -*- coding: utf-8 -*-
"""
实时地图数据模块 - 高德地图API
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import requests


@dataclass
class MapLocation:
    """地图位置信息"""
    lng: float  # 经度
    lat: float  # 纬度
    name: Optional[str] = None  # 地点名称
    address: Optional[str] = None  # 地址


@dataclass
class MapInfo:
    """地图信息数据类"""
    location: MapLocation  # 位置坐标
    zoom: int = 15  # 缩放级别
    location_name: str = ""  # 地点名称
    city: Optional[str] = None  # 城市
    markers: Optional[List[Dict[str, Any]]] = None  # 标记点列表


class AmapMapAPI:
    """高德地图地图API"""
    
    BASE_URL = "https://restapi.amap.com/v3"
    
    def __init__(self, api_key: str):
        """
        初始化高德地图API客户端
        
        Args:
            api_key: 高德地图API密钥
        """
        self.api_key = api_key
    
    def search_location(self, location_name: str, city: Optional[str] = None) -> Optional[MapLocation]:
        """
        根据地点名称搜索位置坐标
        
        Args:
            location_name: 地点名称，例如"中关村"、"天安门"
            city: 城市名称（可选），用于限定搜索范围
            
        Returns:
            MapLocation对象，如果查询失败返回None
        """
        try:
            search_url = f"{self.BASE_URL}/place/text"
            params = {
                "key": self.api_key,
                "keywords": location_name,
                "output": "json",
                "extensions": "base"
            }
            
            if city:
                params["city"] = city
            
            response = requests.get(search_url, params=params, timeout=10)
            data = response.json()
            
            # 检查是否成功
            if data.get("status") != "1":
                print(f"高德API搜索失败: {data.get('info', '未知错误')}")
                return None
            
            pois = data.get("pois", [])
            if not pois:
                print(f"未找到地点: {location_name}")
                return None
            
            # 获取第一个匹配的地点
            poi = pois[0]
            location_str = poi.get("location", "")
            if not location_str:
                return None
            
            # 解析坐标 "经度,纬度"
            coords = location_str.split(",")
            if len(coords) != 2:
                return None
            
            try:
                lng = float(coords[0])
                lat = float(coords[1])
            except ValueError:
                return None
            
            return MapLocation(
                lng=lng,
                lat=lat,
                name=poi.get("name", location_name),
                address=poi.get("address", "")
            )
            
        except Exception as e:
            print(f"搜索地点时发生异常: {e}")
            return None
    
    def get_map_info(self, location_name: str, city: Optional[str] = None, zoom: int = 15) -> Optional[MapInfo]:
        """
        获取完整的地图信息
        
        Args:
            location_name: 地点名称
            city: 城市名称（可选）
            zoom: 缩放级别（默认15）
            
        Returns:
            MapInfo对象，如果查询失败返回None
        """
        location = self.search_location(location_name, city)
        if not location:
            return None
        
        # 构建标记点
        markers = [{
            "lng": location.lng,
            "lat": location.lat,
            "title": location.name or location_name,
            "address": location.address or ""
        }]
        
        return MapInfo(
            location=location,
            zoom=zoom,
            location_name=location.name or location_name,
            city=city,
            markers=markers
        )
    
    def generate_map_url(self, location: MapLocation, zoom: int = 15, size: str = "800*600") -> str:
        """
        生成静态地图URL（用于前端展示）
        
        Args:
            location: 地图位置
            zoom: 缩放级别
            size: 地图尺寸，格式 "宽*高"
            
        Returns:
            静态地图图片URL
        """
        # 高德地图静态地图API
        static_map_url = "https://restapi.amap.com/v3/staticmap"
        params = {
            "key": self.api_key,
            "location": f"{location.lng},{location.lat}",
            "zoom": zoom,
            "size": size,
            "markers": f"mid,,A:{location.lng},{location.lat}",
            "scale": 2  # 高清
        }
        
        # 构建URL
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{static_map_url}?{query_string}"


# 全局客户端实例（单例模式）
_map_client: Optional[AmapMapAPI] = None


def get_map_client(api_key: Optional[str] = None) -> Optional[AmapMapAPI]:
    """
    获取地图API客户端实例（单例）
    
    Args:
        api_key: API密钥（可选，如果不提供则从配置读取）
    
    Returns:
        AmapMapAPI实例，如果未配置API密钥则返回None
    """
    global _map_client
    
    if _map_client is not None:
        return _map_client
    
    from modules.config.settings import AMAP_API_KEY
    
    key = api_key or AMAP_API_KEY
    if not key:
        return None
    
    _map_client = AmapMapAPI(key)
    return _map_client

