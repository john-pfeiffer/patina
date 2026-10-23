#!/usr/bin/env node
// Keep the app version identical everywhere it is declared.
//
//   node scripts/bump-version.mjs 0.2.0     # set the version in every file
//   node scripts/bump-version.mjs --check   # fail if the files disagree (CI)
//
// Files: root package.json, apps/desktop/package.json, tauri.conf.json,
// Cargo.toml + Cargo.lock (the patine-desktop package), sidecar/pyproject.toml,
// sidecar/src/patine/__init__.py.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const arg = process.argv[2];
if (!arg) {
  console.error("usage: bump-version.mjs <x.y.z[-pre]> | --check");
  process.exit(2);
}
const SEMVER = /^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/;

// Each entry: file, regex with the version in group 2 (group 1 is the prefix).
const targets = [
  { file: "package.json", re: /("version":\s*")([^"]+)(")/ },
  { file: "apps/desktop/package.json", re: /("version":\s*")([^"]+)(")/ },
  { file: "apps/desktop/src-tauri/tauri.conf.json", re: /("version":\s*")([^"]+)(")/ },
  { file: "apps/desktop/src-tauri/Cargo.toml", re: /(\[package\][^[]*?\nversion\s*=\s*")([^"]+)(")/ },
  { file: "apps/desktop/src-tauri/Cargo.lock", re: /(name = "patine-desktop"\nversion = ")([^"]+)(")/ },
  { file: "sidecar/pyproject.toml", re: /(\[project\][^[]*?\nversion\s*=\s*")([^"]+)(")/ },
  { file: "sidecar/src/patine/__init__.py", re: /(__version__\s*=\s*")([^"]+)(")/ },
];

const found = targets.map((t) => {
  const text = readFileSync(join(root, t.file), "utf8");
  const m = text.match(t.re);
  if (!m) {
    console.error(`no version found in ${t.file}`);
    process.exit(1);
  }
  return { ...t, text, version: m[2] };
});

if (arg === "--check") {
  const versions = new Set(found.map((f) => f.version));
  for (const f of found) console.log(`${f.version}  ${f.file}`);
  if (versions.size !== 1) {
    console.error("version mismatch across files");
    process.exit(1);
  }
  console.log(`ok: ${[...versions][0]}`);
  process.exit(0);
}

if (!SEMVER.test(arg)) {
  console.error(`not a semver version: ${arg}`);
  process.exit(2);
}
for (const f of found) {
  const next = f.text.replace(f.re, `$1${arg}$3`);
  writeFileSync(join(root, f.file), next);
  console.log(`${f.version} -> ${arg}  ${f.file}`);
}
console.log(`\nNow: git commit -am "Bump version to ${arg}"`);
