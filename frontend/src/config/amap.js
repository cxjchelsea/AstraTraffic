/**
 * 高德地图配置
 * 将你的高德地图 Web端（JS API）Key 配置在这里
 * 
 * 获取方式：
 * 1. 访问 https://console.amap.com/dev/key/app
 * 2. 创建新应用，选择"Web端(JS API)"
 * 3. 复制 Key 到下面的 AMAP_WEB_KEY
 */

// 高德地图 Web 端 API Key（用于前端 JavaScript API）
// 优先从环境变量读取，如果没有则使用默认值
const envKey = import.meta.env.VITE_AMAP_WEB_KEY
export const AMAP_WEB_KEY = envKey && envKey !== 'YOUR_AMAP_WEB_KEY' ? envKey : 'YOUR_AMAP_WEB_KEY'

// 调试：打印配置信息（开发环境）
if (import.meta.env.DEV) {
  console.log('[AMAP Config] VITE_AMAP_WEB_KEY from env:', import.meta.env.VITE_AMAP_WEB_KEY)
  console.log('[AMAP Config] Final AMAP_WEB_KEY:', AMAP_WEB_KEY)
}

// 高德地图 API 版本
export const AMAP_VERSION = '2.0'

// 需要加载的插件
export const AMAP_PLUGINS = [
  'AMap.ToolBar',
  'AMap.Scale',
  'AMap.Geolocation',
  'AMap.Marker',
  'AMap.InfoWindow'
]

