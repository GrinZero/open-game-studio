import assert from "node:assert/strict";
import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { runtimeCopyFilter } from "./runtime-copy.mjs";

test("installed packages include sources and releases but exclude Python envs, caches and credentials", () => {
  const temporary = mkdtempSync(path.join(tmpdir(), "studio-runtime-copy-"));
  try {
    const repo = path.join(temporary, "repo");
    const files = [
      "packages/monster/worker/server.py",
      "packages/monster/dist/release.tar.gz",
      "packages/monster/.env.example",
      "packages/monster/.venv/bin/python",
      "packages/monster/__pycache__/server.pyc",
      "packages/monster/node_modules/dependency/index.js",
      "packages/monster/var/downloaded.glb",
      "packages/monster/endpoint_registry.json",
      "packages/monster/.env",
      "packages/monster/.env.local",
    ];
    for (const name of files) {
      const file = path.join(repo, name);
      mkdirSync(path.dirname(file), { recursive: true });
      writeFileSync(file, "fixture");
    }
    const destination = path.join(temporary, "runtime/packages");
    cpSync(path.join(repo, "packages"), destination, {
      recursive: true,
      filter: (source) => runtimeCopyFilter("packages", source, repo),
    });
    for (const [index, name] of files.entries()) {
      assert.equal(
        existsSync(path.join(temporary, "runtime", name)),
        index < 3,
        name,
      );
    }
  } finally {
    rmSync(temporary, { recursive: true, force: true });
  }
});
