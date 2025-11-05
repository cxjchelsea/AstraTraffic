<template>
  <div class="map-panel">
    <div class="map-header">
      <h3>{{ mapData.location_name || '地图' }}</h3>
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
    // 调试：打印配置信息
    console.log('[MapPanel] AMAP_WEB_KEY:', AMAP_WEB_KEY)
    console.log('[MapPanel] mapData:', props.mapData)
    
    // 检查 API Key 是否配置
    if (!AMAP_WEB_KEY || AMAP_WEB_KEY === 'YOUR_AMAP_WEB_KEY') {
      console.warn('[MapPanel] API Key未配置')
      showError('请配置高德地图 API Key')
      return
    }
    
    // 检查地图数据
    if (!props.mapData || !props.mapData.location) {
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
    
    // 创建地图实例
    mapInstance = new AMap.Map(mapContainer.value, {
      zoom: props.mapData.zoom || 15,
      center: [props.mapData.location.lng, props.mapData.location.lat],
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
    
    // 如果显示路况
    if (props.mapData.show_traffic) {
      const trafficLayer = new AMap.TileLayer.Traffic({
        zIndex: 10,
        autoRefresh: true
      })
      mapInstance.add(trafficLayer)
    }
    
    // 地图加载完成后的回调
    mapInstance.on('complete', () => {
      console.log('地图加载完成')
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
  
  // 更新地图中心点
  mapInstance.setCenter([props.mapData.location.lng, props.mapData.location.lat])
  
  // 更新缩放级别
  if (props.mapData.zoom) {
    mapInstance.setZoom(props.mapData.zoom)
  }
  
  // 更新标记点
  addMarkers(props.mapData.markers || [])
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

