#!/usr/bin/env node
// Bundle the Python sidecar into a single executable with PyInstaller and
// place it where Tauri expects an external binary:
//   apps/desktop/src-tauri/binaries/patine-sidecar-<target-triple>[.exe]
// Release builds then run `tauri build --config src-tauri/tauri.release.conf.json`.
import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const sidecar = join(root, "sidecar");
const shell = process.platform === "win32";

function run(cmd, args, cwd) {
  const r = spawnSync(cmd, args, { cwd, stdio: "inherit", shell });
  if (r.status !== 0) process.exit(r.status ?? 1);
}

const rustc = spawnSync("rustc", ["-vV"], { encoding: "utf8", shell });
const triple = (rustc.stdout.match(/host: (\S+)/) || [])[1];
if (!triple) {
  console.error("could not determine the Rust host triple (is rustc installed?)");
  process.exit(1);
}

run("uv", ["sync", "--extra", "dev"], sidecar);
run("uv", ["run", "pyinstaller", "--noconfirm", "--clean", "patine-sidecar.spec"], sidecar);

const ext = process.platform === "win32" ? ".exe" : "";
const built = join(sidecar, "dist", `patine-sidecar${ext}`);
if (!existsSync(built)) {
  console.error(`expected PyInstaller output at ${built}`);
  process.exit(1);
}
const outDir = join(root, "apps", "desktop", "src-tauri", "binaries");
mkdirSync(outDir, { recursive: true });
const out = join(outDir, `patine-sidecar-${triple}${ext}`);
copyFileSync(built, out);
console.log(`[build-sidecar] ${out}`);
