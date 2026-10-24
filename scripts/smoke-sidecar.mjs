#!/usr/bin/env node
// Launch a built sidecar binary exactly as the shell does, wait for its
// PATINE_READY line, call /health and /theory/parse, then stop it.
//
//   node scripts/smoke-sidecar.mjs                 # auto-locate the binary tauri build produced
//   node scripts/smoke-sidecar.mjs path/to/patine-sidecar
//
// Run after `tauri build` so the binary tested is the signed one inside the
// bundle: on macOS a signing problem (hardened runtime vs the bundled Python
// framework) only shows up there.
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const release = join(root, "apps", "desktop", "src-tauri", "target", "release");
const READY_TIMEOUT_MS = 120_000;
const HEALTH_TIMEOUT_MS = 30_000;

function locate() {
  const candidates =
    process.platform === "darwin"
      ? [join(release, "bundle", "macos", "Patine.app", "Contents", "MacOS", "patine-sidecar"), join(release, "patine-sidecar")]
      : process.platform === "win32"
        ? [join(release, "patine-sidecar.exe")]
        : [join(release, "patine-sidecar")];
  return candidates.find((p) => existsSync(p));
}

const bin = process.argv[2] ?? locate();
if (!bin || !existsSync(bin)) {
  console.error(`sidecar binary not found${bin ? `: ${bin}` : ""}`);
  process.exit(1);
}
console.log(`[smoke] ${bin}`);

const dataDir = mkdtempSync(join(tmpdir(), "patine-smoke-"));
const child = spawn(bin, ["--data-dir", dataDir, "--token", "smoke"], { stdio: ["ignore", "pipe", "pipe"] });
let stdout = "";
let stderr = "";
let done = false;

function finish(code, message) {
  if (done) return;
  done = true;
  console.log(message);
  if (code !== 0) {
    console.error("--- sidecar stdout ---\n" + stdout.trim());
    console.error("--- sidecar stderr (tail) ---\n" + stderr.split("\n").slice(-40).join("\n"));
  }
  try {
    child.kill();
  } catch {}
  setTimeout(() => {
    try {
      rmSync(dataDir, { recursive: true, force: true });
    } catch {}
    process.exit(code);
  }, 300);
}

const readyTimer = setTimeout(() => finish(1, `[smoke] FAIL: no PATINE_READY within ${READY_TIMEOUT_MS / 1000}s`), READY_TIMEOUT_MS);

async function probe(info) {
  const headers = { "X-Patine-Token": info.token };
  const deadline = Date.now() + HEALTH_TIMEOUT_MS;
  let health = null;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`${info.url}/health`);
      if (r.ok) {
        health = await r.json();
        break;
      }
    } catch {}
    await new Promise((r) => setTimeout(r, 250));
  }
  if (!health) return finish(1, "[smoke] FAIL: /health never answered");
  const parse = await fetch(`${info.url}/theory/parse`, {
    method: "POST",
    headers: { ...headers, "content-type": "application/json" },
    body: JSON.stringify({ text: "i iv V", key: "C minor" }),
  }).then((r) => r.json());
  if (!parse.ok || parse.chords?.length !== 3) return finish(1, `[smoke] FAIL: /theory/parse gave ${JSON.stringify(parse)}`);
  finish(0, `[smoke] OK: ${health.name} ${health.version} at ${info.url}, theory parse ok`);
}

child.stdout.on("data", (d) => {
  stdout += d.toString();
  const m = stdout.match(/^PATINE_READY (.*)$/m);
  if (m && !readyTimer._probed) {
    readyTimer._probed = true;
    clearTimeout(readyTimer);
    let info;
    try {
      info = JSON.parse(m[1]);
    } catch (e) {
      return finish(1, `[smoke] FAIL: bad ready line: ${m[1]}`);
    }
    void probe(info);
  }
});
child.stderr.on("data", (d) => {
  stderr += d.toString();
});
child.on("error", (e) => finish(1, `[smoke] FAIL: could not spawn: ${e.message}`));
child.on("exit", (code, signal) => {
  if (!done) finish(1, `[smoke] FAIL: sidecar exited (code ${code}, signal ${signal}) before it was ready`);
});
