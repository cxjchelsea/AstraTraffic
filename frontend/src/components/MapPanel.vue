<template>
  <div class="map-panel">
    <div class="map-header">
      <h3>{{ getHeaderTitle() }}</h3>
      <button @click="$emit('close-map')" class="close-button">×</button>
    </div>
    <div class="map-container" ref="mapContainer"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, nextTick } from 'vue'
import { AMAP_WEB_KEY, AMAP_VERSION, AMAP_PLUGINS } from '../config/amap'

const props = defineProps({
  mapData: {
    type: Object,
    required: true
  }
})

defineEmits(['close-map'])

const mapContainer = ref(null)
let mapInstance = null
let markers = []
let trafficLayer = null  // 路况图层实例
let routePolyline = null  // 路径折线实例

onMounted(() => {
  nextTick(() => {
    initMap()
  })
})

watch(() => props.mapData, () => {
  if (mapInstance && props.mapData) {
    updateMap()
  }
}, { deep: true })

const initMap = async () => {
  try {
    // 调试：仅在开发环境打印配置信息
    if (import.meta.env.DEV) {
      console.log('[MapPanel] AMAP_WEB_KEY:', AMAP_WEB_KEY)
      console.log('[MapPanel] mapData:', props.mapData)
    }
    
    // 检查 API Key 是否配置
    if (!AMAP_WEB_KEY || AMAP_WEB_KEY === 'YOUR_AMAP_WEB_KEY') {
      console.warn('[MapPanel] API Key未配置')
      showError('请配置高德地图 API Key')
      return
    }
    
    // 检查地图数据（支持路径数据或地图数据）
    if (!props.mapData || (!props.mapData.location && !props.mapData.route_data)) {
      console.warn('[MapPanel] 地图数据不完整:', props.mapData)
      showError('地图数据不完整')
      return
    }
    
    // 动态加载高德地图API
    const AMapLoader = await import('@amap/amap-jsapi-loader')
    
    const loader = AMapLoader.default || AMapLoader
    
    const AMap = await loader.load({
      key: AMAP_WEB_KEY,
      version: AMAP_VERSION,
      plugins: AMAP_PLUGINS
    })
    
    // 确定初始中心点和缩放级别
    let initialCenter = [116.397428, 39.90923]  // 默认北京天安门
    let initialZoom = 15
    
    if (props.mapData.route_data) {
      // 如果有路径数据，使用起点作为中心
      initialCenter = [props.mapData.route_data.origin.lng, props.mapData.route_data.origin.lat]
    } else if (props.mapData.location) {
      // 如果有地图数据，使用地图位置
      initialCenter = [props.mapData.location.lng, props.mapData.location.lat]
      initialZoom = props.mapData.zoom || 15
    }
    
    // 创建地图实例
    mapInstance = new AMap.Map(mapContainer.value, {
      zoom: initialZoom,
      center: initialCenter,
      viewMode: '3D',
      mapStyle: 'amap://styles/normal'  // 标准地图样式
    })
    
    // 添加工具条和比例尺
    mapInstance.addControl(new AMap.ToolBar({
      position: 'RT'  // 右上角
    }))
    mapInstance.addControl(new AMap.Scale({
      position: 'LB'  // 左下角
    }))
    
    // 添加标记点
    addMarkers(props.mapData.markers || [])
    
    // 如果有路径数据，绘制路径
    if (props.mapData.route_data) {
      drawRoute(props.mapData.route_data)
    }
    
    // 如果显示路况
    if (props.mapData.show_traffic) {
      trafficLayer = new AMap.TileLayer.Traffic({
        zIndex: 10,
        autoRefresh: true
      })
      mapInstance.add(trafficLayer)
      if (import.meta.env.DEV) {
        console.log('[MapPanel] 已启用实时路况图层')
      }
    }
    
    // 如果有边界信息，使用边界自动调整视野（优先于缩放级别）
    if (props.mapData.bounds) {
      const bounds = props.mapData.bounds
      const bounds_obj = new AMap.Bounds(
        [bounds.southwest.lng, bounds.southwest.lat],
        [bounds.northeast.lng, bounds.northeast.lat]
      )
      // 调整地图视野以完整显示边界区域
      mapInstance.setBounds(bounds_obj)
      if (import.meta.env.DEV) {
        console.log('[MapPanel] 使用边界自动调整视野:', bounds)
      }
    } else if (props.mapData.route_data) {
      // 如果有路径数据，调整视野以包含整个路径（在drawRoute中处理）
      // 这里不需要额外操作
    } else if (props.mapData.zoom && props.mapData.location) {
      // 如果没有边界，使用指定的缩放级别
      mapInstance.setZoom(props.mapData.zoom)
      mapInstance.setCenter([props.mapData.location.lng, props.mapData.location.lat])
    }
    
    // 地图加载完成后的回调
    mapInstance.on('complete', () => {
      if (import.meta.env.DEV) {
        console.log('地图加载完成')
      }
    })
    
  } catch (error) {
    console.error('地图初始化失败:', error)
    showError(`地图加载失败: ${error.message || '未知错误'}`)
  }
}

const addMarkers = (markerList) => {
  // 清除旧标记
  clearMarkers()
  
  if (!mapInstance || !markerList || markerList.length === 0) {
    return
  }
  
  markerList.forEach(markerData => {
    const marker = new AMap.Marker({
      position: [markerData.lng, markerData.lat],
      title: markerData.title || markerData.address || '',
      map: mapInstance
    })
    
    // 如果有标题或地址，添加信息窗口
    if (markerData.title || markerData.address) {
      const infoWindow = new AMap.InfoWindow({
        content: `<div style="padding: 0.5rem;">
          <strong>${markerData.title || '位置'}</strong><br/>
          ${markerData.address || ''}
        </div>`,
        offset: new AMap.Pixel(0, -30)
      })
      
      marker.on('click', () => {
        infoWindow.open(mapInstance, marker.getPosition())
      })
    }
    
    markers.push(marker)
  })
  
  // 如果只有一个标记点，自动调整视野
  if (markers.length === 1) {
    mapInstance.setZoom(props.mapData.zoom || 15)
    mapInstance.setCenter(markers[0].getPosition())
  } else if (markers.length > 1) {
    // 多个标记点，自动调整视野以包含所有标记
    mapInstance.setFitView(markers)
  }
}

const clearMarkers = () => {
  markers.forEach(marker => {
    marker.setMap(null)
  })
  markers = []
  // 同时清除路径
  clearRoute()
}

const getHeaderTitle = () => {
  if (props.mapData?.route_data) {
    return '路径规划'
  }
  return props.mapData?.location_name || '地图'
}

const showError = (message) => {
  if (mapContainer.value) {
    mapContainer.value.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #999; flex-direction: column; padding: 2rem;">
        <div style="text-align: center;">
          <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">${message}</p>
          <p style="font-size: 0.8rem; margin-top: 0.5rem; color: #666;">
            请检查高德地图 API Key 配置<br/>
            配置文件：frontend/src/config/amap.js
          </p>
        </div>
      </div>
    `
  }
}

const updateMap = () => {
  if (!mapInstance || !props.mapData) return
  
  // 如果有边界信息，优先使用边界自动调整视野
  if (props.mapData.bounds) {
    const bounds = props.mapData.bounds
    const bounds_obj = new AMap.Bounds(
      [bounds.southwest.lng, bounds.southwest.lat],
      [bounds.northeast.lng, bounds.northeast.lat]
    )
    mapInstance.setBounds(bounds_obj)
    if (import.meta.env.DEV) {
      console.log('[MapPanel] 更新地图视野（使用边界）:', bounds)
    }
  } else {
    // 更新地图中心点
    mapInstance.setCenter([props.mapData.location.lng, props.mapData.location.lat])
    
    // 更新缩放级别
    if (props.mapData.zoom) {
      mapInstance.setZoom(props.mapData.zoom)
    }
  }
  
  // 更新路况图层
  if (props.mapData.show_traffic) {
    // 如果还没有路况图层，则添加
    if (!trafficLayer) {
      trafficLayer = new AMap.TileLayer.Traffic({
        zIndex: 10,
        autoRefresh: true
      })
      mapInstance.add(trafficLayer)
      if (import.meta.env.DEV) {
        console.log('[MapPanel] 已启用实时路况图层')
      }
    }
    // 如果已有路况图层，它会自动刷新（autoRefresh: true）
  } else {
    // 如果不需要显示路况，则移除图层
    if (trafficLayer) {
      mapInstance.remove(trafficLayer)
      trafficLayer = null
      if (import.meta.env.DEV) {
        console.log('[MapPanel] 已移除路况图层')
      }
    }
  }
  
  // 更新标记点
  addMarkers(props.mapData.markers || [])
  
  // 如果有路径数据，绘制路径
  if (props.mapData.route_data) {
    drawRoute(props.mapData.route_data)
  }
}

const drawRoute = (routeData) => {
  // 清除旧路径
  clearRoute()
  
  if (import.meta.env.DEV) {
    console.log('[MapPanel] drawRoute 被调用，routeData:', routeData)
    console.log('[MapPanel] routeData 类型:', typeof routeData)
    console.log('[MapPanel] routeData 键:', routeData ? Object.keys(routeData) : 'N/A')
    console.log('[MapPanel] mapInstance:', !!mapInstance)
    console.log('[MapPanel] routeData.polyline:', routeData?.polyline)
    console.log('[MapPanel] routeData.origin:', routeData?.origin)
    console.log('[MapPanel] routeData.destination:', routeData?.destination)
  }
  
  if (!mapInstance || !routeData || !routeData.polyline) {
    if (import.meta.env.DEV) {
      console.warn('[MapPanel] 无法绘制路径: 缺少必要数据', { 
        mapInstance: !!mapInstance, 
        routeData: routeData,
        hasPolyline: !!routeData?.polyline,
        polyline: routeData?.polyline,
        routeDataKeys: routeData ? Object.keys(routeData) : []
      })
    }
    return
  }
  
  try {
    if (import.meta.env.DEV) {
      console.log('[MapPanel] 开始绘制路径:', { 
        origin: routeData.origin, 
        destination: routeData.destination,
        polylineLength: routeData.polyline?.length,
        polylinePreview: routeData.polyline?.substring(0, 100)
      })
    }
    
    // 解析路径坐标串（高德地图格式：经度,纬度;经度,纬度;...）
    // 注意：高德API返回的polyline可能是编码后的，需要先解码
    let polylineStr = routeData.polyline
    
    // 检查是否是编码格式（编码后的字符串通常不包含分号）
    if (!polylineStr.includes(';') && !polylineStr.includes(',')) {
      // 可能是编码格式，需要解码（使用高德地图JS API的解码方法）
      if (window.AMap && window.AMap.GeometryUtil) {
        // 使用高德地图JS API的解码方法
        const decoded = window.AMap.GeometryUtil.decodeLine(polylineStr)
        if (decoded && decoded.length > 0) {
          const path = decoded.map(point => [point.lng, point.lat])
          _drawPolylineWithMarkers(path, routeData)
          return
        }
      }
      console.error('[MapPanel] polyline格式异常，无法解析:', polylineStr.substring(0, 50))
      return
    }
    
    // 未编码格式：直接解析
    const path = polylineStr.split(';').map(point => {
      const [lng, lat] = point.split(',')
      return [parseFloat(lng), parseFloat(lat)]
    }).filter(point => !isNaN(point[0]) && !isNaN(point[1]))
    
    if (path.length === 0) {
      console.error('[MapPanel] 解析后的路径点为空')
      return
    }
    
    _drawPolylineWithMarkers(path, routeData)
    
  } catch (error) {
    console.error('[MapPanel] 绘制路径失败:', error, routeData)
  }
}

const _drawPolylineWithMarkers = (path, routeData) => {
  // 创建路径折线
  routePolyline = new AMap.Polyline({
    path: path,
    isOutline: true,
    outlineColor: '#ffeeff',
    borderWeight: 3,
    strokeColor: '#3366FF',
    strokeOpacity: 1,
    strokeWeight: 6,
    lineJoin: 'round',
    lineCap: 'round',
    zIndex: 50,
    map: mapInstance
  })
  
  // 添加起点和终点标记
  if (routeData.origin && routeData.origin.lng && routeData.origin.lat) {
    const originMarker = new AMap.Marker({
      position: [routeData.origin.lng, routeData.origin.lat],
      icon: new AMap.Icon({
        size: new AMap.Size(32, 32),
        image: 'https://webapi.amap.com/theme/v1.3/markers/n/start.png',
        imageSize: new AMap.Size(32, 32)
      }),
      map: mapInstance
    })
    markers.push(originMarker)
  }
  
  if (routeData.destination && routeData.destination.lng && routeData.destination.lat) {
    const destMarker = new AMap.Marker({
      position: [routeData.destination.lng, routeData.destination.lat],
      icon: new AMap.Icon({
        size: new AMap.Size(32, 32),
        image: 'https://webapi.amap.com/theme/v1.3/markers/n/end.png',
        imageSize: new AMap.Size(32, 32)
      }),
      map: mapInstance
    })
    markers.push(destMarker)
  }
  
  // 调整地图视野以包含整个路径
  if (routePolyline) {
    mapInstance.setFitView([routePolyline], false, [50, 50, 50, 50])
  }
  
  if (import.meta.env.DEV) {
    console.log('[MapPanel] 已绘制路径:', { pathLength: path.length, routeData })
  }
}

const clearRoute = () => {
  if (routePolyline) {
    routePolyline.setMap(null)
    routePolyline = null
  }
}
</script>

<style scoped>
.map-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: white;
  border-left: 1px solid #e0e0e0;
}

.map-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border-bottom: 1px solid #e0e0e0;
  background: #fafafa;
}

.map-header h3 {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: #333;
}

.close-button {
  width: 2rem;
  height: 2rem;
  border: none;
  background: transparent;
  font-size: 1.5rem;
  color: #999;
  cursor: pointer;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.2s;
}

.close-button:hover {
  background: #f0f0f0;
}

.map-container {
  flex: 1;
  width: 100%;
}
</style>

