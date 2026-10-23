import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

// Port, strictPort, clearScreen and envPrefix are what the Tauri shell expects
// (src-tauri/tauri.conf.json points `devUrl` at 1420).
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  envPrefix: ['VITE_', 'TAURI_'],
  server: {
    port: 1420,
    strictPort: true,
    host: '127.0.0.1',
  },
  build: {
    outDir: 'dist',
    target: 'es2022',
    sourcemap: true,
  },
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
});
