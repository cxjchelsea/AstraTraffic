<template>
  <div class="app-container">
    <header class="app-header">
      <h1>🚦 AstraTraffic - 智能出行管家</h1>
    </header>
    
    <main class="app-main">
      <ChatPanel 
        :messages="messages" 
        :loading="loading"
        @send-message="handleSendMessage"
      />
      
      <MapPanel 
        v-if="mapData" 
        :map-data="mapData"
        @close-map="closeMap"
      />
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import ChatPanel from './components/ChatPanel.vue'
import MapPanel from './components/MapPanel.vue'
import { sendChatMessage } from './api/chat'

const messages = ref([])
const loading = ref(false)
const mapData = ref(null)

const handleSendMessage = async (query) => {
  // 添加用户消息
  messages.value.push({
    role: 'user',
    content: query,
    timestamp: new Date()
  })
  
  loading.value = true
  
  // 添加一个"处理中"的提示消息（如果处理时间较长）
  let processingMessageId = null
  const processingTimer = setTimeout(() => {
    processingMessageId = messages.value.length
    messages.value.push({
      role: 'assistant',
      content: '正在处理您的请求，请稍候...（这可能需要一些时间）',
      timestamp: new Date(),
      isProcessing: true
    })
  }, 3000)  // 3秒后显示处理中提示
  
  try {
    const response = await sendChatMessage(query)
    
    // 清除处理中提示
    if (processingMessageId !== null) {
      messages.value.splice(processingMessageId, 1)
    }
    clearTimeout(processingTimer)
    
    // 添加AI回复
    messages.value.push({
      role: 'assistant',
      content: response.answer,
      intent: response.intent,
      hits: response.hits,
      timestamp: new Date()
    })
    
    // 如果有地图数据，显示地图
    if (response.map_data) {
      if (import.meta.env.DEV) {
        console.log('[App] 收到地图数据:', response.map_data)
      }
      mapData.value = response.map_data
    } else {
      if (import.meta.env.DEV) {
        console.log('[App] 响应中没有地图数据')
        console.log('[App] 完整响应:', response)
      }
    }
  } catch (error) {
    // 清除处理中提示
    if (processingMessageId !== null) {
      messages.value.splice(processingMessageId, 1)
    }
    clearTimeout(processingTimer)
    
    console.error('发送消息失败:', error)
    const errorMessage = error.message || '抱歉，处理您的请求时发生错误。请稍后重试。'
    messages.value.push({
      role: 'error',
      content: errorMessage,
      timestamp: new Date()
    })
  } finally {
    loading.value = false
  }
}

const closeMap = () => {
  mapData.value = null
}
</script>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: #f5f5f5;
}

.app-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1rem 2rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.app-header h1 {
  font-size: 1.5rem;
  font-weight: 600;
}

.app-main {
  flex: 1;
  display: flex;
  overflow: hidden;
}

.app-main > * {
  flex: 1;
}

@media (max-width: 768px) {
  .app-main {
    flex-direction: column;
  }
}
</style>

