import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/login": "http://localhost:3000",
      "/register": "http://localhost:3000",
      "/api": "http://localhost:3000",
      "/logout": "http://localhost:3000",
    },
  },
})