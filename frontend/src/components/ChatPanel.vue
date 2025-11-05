<template>
  <div class="chat-panel">
    <div class="chat-messages" ref="messagesContainer">
      <div
        v-for="(message, index) in messages"
        :key="index"
        :class="['message', `message-${message.role}`]"
      >
        <div class="message-content">
          <div class="message-text">{{ message.content }}</div>
          
          <div v-if="message.intent" class="message-meta">
            <span class="intent-badge">意图: {{ message.intent }}</span>
          </div>
          
          <div v-if="message.hits && message.hits.length > 0" class="message-hits">
            <details>
              <summary>参考来源 ({{ message.hits.length }})</summary>
              <ul>
                <li v-for="(hit, idx) in message.hits" :key="idx">
                  <span class="hit-score">[{{ hit.score.toFixed(2) }}]</span>
                  {{ hit.text.substring(0, 100) }}{{ hit.text.length > 100 ? '...' : '' }}
                  <span class="hit-source">({{ hit.source }})</span>
                </li>
              </ul>
            </details>
          </div>
        </div>
        <div class="message-time">
          {{ formatTime(message.timestamp) }}
        </div>
      </div>
      
      <div v-if="loading" class="message message-assistant">
        <div class="message-content">
          <div class="loading-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      </div>
    </div>
    
    <div class="chat-input-container">
      <input
        v-model="inputText"
        type="text"
        placeholder="输入您的问题..."
        @keyup.enter="handleSend"
        :disabled="loading"
        class="chat-input"
      />
      <button
        @click="handleSend"
        :disabled="loading || !inputText.trim()"
        class="send-button"
      >
        发送
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'

const props = defineProps({
  messages: {
    type: Array,
    required: true
  },
  loading: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['send-message'])

const inputText = ref('')
const messagesContainer = ref(null)

const handleSend = () => {
  if (!inputText.value.trim() || props.loading) {
    return
  }
  
  emit('send-message', inputText.value.trim())
  inputText.value = ''
}

const formatTime = (timestamp) => {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { 
    hour: '2-digit', 
    minute: '2-digit' 
  })
}

// 自动滚动到底部
watch(() => props.messages.length, () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
})

watch(() => props.loading, () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
})
</script>

<style scoped>
.chat-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: white;
  border-right: 1px solid #e0e0e0;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.message {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  animation: fadeIn 0.3s ease-in;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message-user {
  align-items: flex-end;
}

.message-assistant,
.message-error {
  align-items: flex-start;
}

.message-content {
  max-width: 70%;
  padding: 0.75rem 1rem;
  border-radius: 1rem;
  word-wrap: break-word;
}

.message-user .message-content {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-bottom-right-radius: 0.25rem;
}

.message-assistant .message-content {
  background: #f0f0f0;
  color: #333;
  border-bottom-left-radius: 0.25rem;
}

.message-error .message-content {
  background: #ffebee;
  color: #c62828;
  border: 1px solid #ef5350;
}

.message-text {
  line-height: 1.5;
  white-space: pre-wrap;
}

.message-meta {
  margin-top: 0.5rem;
  font-size: 0.75rem;
}

.intent-badge {
  display: inline-block;
  padding: 0.25rem 0.5rem;
  background: rgba(0, 0, 0, 0.1);
  border-radius: 0.25rem;
  font-size: 0.7rem;
}

.message-hits {
  margin-top: 0.5rem;
  font-size: 0.75rem;
}

.message-hits details {
  cursor: pointer;
}

.message-hits summary {
  padding: 0.25rem;
  color: #666;
}

.message-hits ul {
  margin-top: 0.5rem;
  padding-left: 1.5rem;
  list-style: none;
}

.message-hits li {
  padding: 0.25rem 0;
  border-bottom: 1px solid #e0e0e0;
}

.hit-score {
  color: #667eea;
  font-weight: 600;
  margin-right: 0.5rem;
}

.hit-source {
  color: #999;
  font-size: 0.7rem;
  margin-left: 0.5rem;
}

.message-time {
  font-size: 0.7rem;
  color: #999;
  padding: 0 0.5rem;
}

.loading-indicator {
  display: flex;
  gap: 0.5rem;
  padding: 0.5rem;
}

.loading-indicator span {
  width: 8px;
  height: 8px;
  background: #667eea;
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out both;
}

.loading-indicator span:nth-child(1) {
  animation-delay: -0.32s;
}

.loading-indicator span:nth-child(2) {
  animation-delay: -0.16s;
}

@keyframes bounce {
  0%, 80%, 100% {
    transform: scale(0);
  }
  40% {
    transform: scale(1);
  }
}

.chat-input-container {
  display: flex;
  padding: 1rem;
  border-top: 1px solid #e0e0e0;
  background: white;
  gap: 0.5rem;
}

.chat-input {
  flex: 1;
  padding: 0.75rem 1rem;
  border: 1px solid #e0e0e0;
  border-radius: 1.5rem;
  font-size: 1rem;
  outline: none;
  transition: border-color 0.2s;
}

.chat-input:focus {
  border-color: #667eea;
}

.chat-input:disabled {
  background: #f5f5f5;
  cursor: not-allowed;
}

.send-button {
  padding: 0.75rem 1.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 1.5rem;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: opacity 0.2s;
}

.send-button:hover:not(:disabled) {
  opacity: 0.9;
}

.send-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>

