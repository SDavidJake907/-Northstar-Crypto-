import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      '/api/health': {
        target: 'http://127.0.0.1:9091',
        changeOrigin: true,
        rewrite: () => '/health',
      },
      '/api/chat': {
        target: 'http://127.0.0.1:9090',
        changeOrigin: true,
        rewrite: () => '/chat',
      },
      '/api/ping/entry': {
        target: 'http://127.0.0.1:8081',
        changeOrigin: true,
        rewrite: () => '/health',
      },
      '/api/exit': {
        target: 'http://127.0.0.1:8082',
        changeOrigin: true,
        rewrite: () => '/v1/chat/completions',
      },
      '/api/ping/exit': {
        target: 'http://127.0.0.1:8082',
        changeOrigin: true,
        rewrite: () => '/health',
      },
      '/api/atlas': {
        target: 'http://127.0.0.1:8083',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/atlas/, ''),
      },
      '/api/telemetry': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
        ws: true,
        rewrite: () => '/',
      },
      '/api/npu': {
        target: 'http://127.0.0.1:8084',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/npu/, ''),
      },
    },
  },
})
