const { defineConfig, devices } = require("@playwright/test");
const fs = require("fs");
const path = require("path");

const workspacePython = path.resolve(".venv", "Scripts", "python.exe");
const pythonCommand = process.env.PLAYWRIGHT_PYTHON
  || (process.platform === "win32" && fs.existsSync(workspacePython)
    ? JSON.stringify(workspacePython)
    : process.platform === "win32" ? "py -3" : "python3");

module.exports = defineConfig({
  testDir: "./tests/browser",
  outputDir: "./outputs/playwright",
  snapshotPathTemplate: "{testDir}/__screenshots__/{arg}{ext}",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  reporter: "list",
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || "http://127.0.0.1:5055",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: process.env.PLAYWRIGHT_BASE_URL ? undefined : {
    command: `${pythonCommand} tests/ui_fixture_server.py`,
    url: "http://127.0.0.1:5055",
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],
});
