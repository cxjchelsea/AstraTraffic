import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig(({ mode }) => {
  // 获取项目根目录（frontend目录的父目录）
  // 使用 fileURLToPath 和 URL 来兼容 ES 模块
  const __dirname = fileURLToPath(new URL('.', import.meta.url))
  const root = fileURLToPath(new URL('..', import.meta.url))  // 项目根目录
  
  return {
    plugins: [vue()],
    server: {
      port: 3000,
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8000',  // 使用IPv4地址，避免IPv6连接问题
          changeOrigin: true,
          rewrite: (path) => path  // 保持路径不变
        }
      }
    },
    // 指定环境变量目录为项目根目录
    // 这样可以在项目根目录的 .env 文件中配置 VITE_AMAP_WEB_KEY
    envDir: root,
    envPrefix: 'VITE_',
  }
})

