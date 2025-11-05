import axios from 'axios'

const API_BASE_URL = '/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,  // 增加到120秒（2分钟），因为RAG处理可能需要较长时间
  headers: {
    'Content-Type': 'application/json'
  }
})

/**
 * 发送聊天消息
 * @param {string} query - 用户查询
 * @param {string} sessionId - 会话ID（可选）
 * @returns {Promise} API响应
 */
export async function sendChatMessage(query, sessionId = 'default') {
  try {
    const response = await api.post('/chat', {
      query,
      session_id: sessionId
    })
    return response.data
  } catch (error) {
    console.error('API请求失败:', error)
    
    // 提取更友好的错误信息
    if (error.response) {
      // 服务器返回了错误响应
      const errorMessage = error.response.data?.detail || error.response.data?.message || '服务器错误'
      throw new Error(errorMessage)
    } else if (error.request) {
      // 请求已发出但没有收到响应
      throw new Error('无法连接到服务器，请检查后端服务是否运行')
    } else {
      // 其他错误
      throw new Error(error.message || '请求失败')
    }
  }
}

/**
 * 健康检查
 * @returns {Promise} API响应
 */
export async function healthCheck() {
  try {
    const response = await api.get('/health')
    return response.data
  } catch (error) {
    console.error('健康检查失败:', error)
    throw error
  }
}

