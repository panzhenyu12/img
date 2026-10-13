import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// SharedArrayBuffer permits local ORT worker threads. Production hosting must
// send the same headers; browsers without isolation use one local thread.
const isolationHeaders = {'Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp'};

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '127.0.0.1',
    headers: isolationHeaders,
  },
  preview: {headers: isolationHeaders},
});
