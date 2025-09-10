import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist', // Ensure output directory is 'dist'
    assetsDir: 'assets', // Ensure assets are placed in 'assets' subdirectory within dist
  },
})
