import { defineConfig } from '@playwright/test'

// Runs against the dev servers: `../scripts/dev.sh` (API :8000, explorer :5173).
export default defineConfig({
  testDir: 'e2e',
  timeout: 600_000,
  use: {
    baseURL: 'http://localhost:5173',
    viewport: { width: 1600, height: 950 },
    launchOptions: {
      executablePath: process.env.CHROMIUM_PATH,
      args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    },
  },
  webServer: [
    { command: 'cd .. && uv run uvicorn catlab.api:app --port 8000', url: 'http://127.0.0.1:8000/api/health', reuseExistingServer: true, timeout: 120_000 },
    { command: 'npm run dev', url: 'http://localhost:5173', reuseExistingServer: true },
  ],
})
