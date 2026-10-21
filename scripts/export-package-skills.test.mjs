import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { exportPackageSkills } from "./export-package-skills.mjs";

function fixture(run) {
  const repo = mkdtempSync(path.join(tmpdir(), "studio-package-skills-"));
  try {
    const source = path.join(repo, "packages/monster/skills/monster-workflow");
    mkdirSync(path.join(source, "references"), { recursive: true });
    writeFileSync(path.join(source, "SKILL.md"), "[SOP](references/sop.md)\n");
    writeFileSync(path.join(source, "references/sop.md"), "Review material\n");
    run(repo, source, path.join(repo, "skills/monster-workflow"));
  } finally {
    rmSync(repo, { recursive: true, force: true });
  }
}

test("package skills export with self-contained references and update without stale files", () => {
  fixture((repo, source, target) => {
    exportPackageSkills(repo);
    assert.equal(readFileSync(path.join(target, "references/sop.md"), "utf8"), "Review material\n");
    writeFileSync(path.join(source, "references/sop.md"), "Updated review material\n");
    writeFileSync(path.join(target, "references/stale.md"), "old generated file\n");
    exportPackageSkills(repo);
    assert.equal(readFileSync(path.join(target, "references/sop.md"), "utf8"), "Updated review material\n");
    assert.equal(existsSync(path.join(target, "references/stale.md")), false);
    assert.equal(JSON.parse(readFileSync(path.join(target, ".package-skill.json"))).source,
      "packages/monster/skills/monster-workflow");
  });
});

test("package export preserves a hand-maintained skill on name collision", () => {
  fixture((repo, _source, target) => {
    mkdirSync(target, { recursive: true });
    writeFileSync(path.join(target, "SKILL.md"), "hand-maintained\n");
    assert.throws(() => exportPackageSkills(repo), /conflicts with an existing skill/);
    assert.equal(readFileSync(path.join(target, "SKILL.md"), "utf8"), "hand-maintained\n");
  });
});
