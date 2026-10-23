#!/usr/bin/env node
// One command for development: `pnpm dev`.
//
// Checks the toolchain, syncs the sidecar's Python environment, then runs
// `tauri dev`, which starts Vite with hot reload and the Rust shell. The
// shell spawns the sidecar from source (`uv run patine-sidecar --dev`) with
// auto-reload, so a change anywhere is live without a packaged build.
import { spawnSync, spawn } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

function has(cmd, args = ["--version"]) {
  const r = spawnSync(cmd, args, { stdio: "ignore", shell: process.platform === "win32" });
  return r.status === 0;
}

const missing = [];
if (!has("uv")) missing.push("uv (https://docs.astral.sh/uv/)");
if (!has("cargo")) missing.push("cargo (https://rustup.rs)");
if (!has("pnpm")) missing.push("pnpm");
if (missing.length) {
  console.error("Patine dev needs:\n  - " + missing.join("\n  - "));
  process.exit(1);
}

console.log("[dev] syncing sidecar environment (uv sync)…");
const sync = spawnSync("uv", ["sync", "--extra", "dev"], {
  cwd: join(root, "sidecar"),
  stdio: "inherit",
  shell: process.platform === "win32",
});
if (sync.status !== 0) process.exit(sync.status ?? 1);

if (!existsSync(join(root, "node_modules"))) {
  console.log("[dev] installing node modules…");
  const inst = spawnSync("pnpm", ["install"], { cwd: root, stdio: "inherit", shell: process.platform === "win32" });
  if (inst.status !== 0) process.exit(inst.status ?? 1);
}

// Tauri reads frontendDist at compile time even in dev; keep the folder present.
mkdirSync(join(root, "apps", "desktop", "dist"), { recursive: true });

console.log("[dev] launching desktop shell (tauri dev)…");
const child = spawn("pnpm", ["--filter", "@patine/desktop", "dev"], {
  cwd: root,
  stdio: "inherit",
  shell: process.platform === "win32",
  env: { ...process.env, RUST_LOG: process.env.RUST_LOG ?? "info" },
});
child.on("exit", (code) => process.exit(code ?? 0));
