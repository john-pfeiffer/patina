import { playwright } from '@vitest/browser-playwright';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

// Web Audio needs a real browser: the playback engine is rendered offline in
// headless Chromium. `pnpm test:browser`; CI installs Chromium first.
// PATINE_CHROMIUM points at a Chromium binary when Playwright's own download
// is not wanted.
export default defineConfig({
  plugins: [react()],
  test: {
    include: ['src/**/*.browser.test.ts'],
    browser: {
      enabled: true,
      headless: true,
      provider: playwright({
        launchOptions: {
          executablePath: process.env.PATINE_CHROMIUM || undefined,
          args: ['--autoplay-policy=no-user-gesture-required'],
        },
      }),
      instances: [{ browser: 'chromium' }],
    },
  },
});
