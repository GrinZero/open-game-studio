import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { exportPackageSkills } from "./export-package-skills.mjs";

exportPackageSkills();

const locales = [
  "README.md",
  "README.zh-CN.md",
  "README.zh-TW.md",
  "README.ja.md",
  "README.ko.md",
];
const read = (file) => readFileSync(file, "utf8");
const withoutCode = (text) => text.replace(/```[\s\S]*?```/g, "");
const targets = (text) =>
  [
    ...withoutCode(text).matchAll(
      /\]\(([^\s)]+)(?:\s+"[^"]*")?\)|(?:href|src)="([^"]+)"/g,
    ),
  ].map((m) => m[1] || m[2]);
const anchors = (text) => {
  const body = withoutCode(text);
  return new Set([
    ...[...body.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]),
    ...[...body.matchAll(/^#{1,6}\s+(.+)$/gm)].map((m) =>
      m[1]
        .toLowerCase()
        .replace(/[^\p{L}\p{N}_\-\s]/gu, "")
        .replace(/ /g, "-"),
    ),
  ]);
};
function verifyLinks(file) {
  for (const target of targets(read(file))) {
    if (/^(?:https?:|mailto:|data:)/.test(target)) continue;
    const [relative, fragment] = target.split("#");
    const destination = relative
      ? path.resolve(path.dirname(file), decodeURIComponent(relative))
      : path.resolve(file);
    assert.ok(existsSync(destination), `${file}: missing ${target}`);
    if (fragment && destination.endsWith(".md"))
      assert.ok(
        anchors(read(destination)).has(decodeURIComponent(fragment)),
        `${file}: missing anchor ${target}`,
      );
  }
}
function shape(text) {
  return {
    headings: [...text.matchAll(/^#{1,6}\s/gm)].map((m) => m[0].trim().length),
    anchors: [...text.matchAll(/<a id="([^"]+)"><\/a>/g)].map((m) => m[1]),
    tables: (text.match(/(?:^\|.*\|\n?)+/gm) || []).map((table) =>
      table
        .trim()
        .split("\n")
        .map((row) => row.split("|").length - 2),
    ),
    links: targets(text),
    commands: [...text.matchAll(/```sh\n([\s\S]*?)```/g)].map((m) => m[1]),
    images: (text.match(/!\[[^\]]*\]\(/g) || []).length,
  };
}
const baseline = shape(read(locales[0]));
const source = read("src/server/operations.ts");
const names = [
  ...source
    .slice(0, source.indexOf("export function dispatch"))
    .matchAll(/^  ([a-z_]+): \{/gm),
].map((m) => m[1]);
const skillCount = readdirSync("skills").filter((name) =>
  existsSync(`skills/${name}/SKILL.md`),
).length;
assert.equal(
  names.length,
  26,
  "Domain tool count changed; update README facts and tool index",
);
assert.equal(skillCount, 11, "Skill count changed; update README facts");
const contract = read("docs/TOOL_CONTRACT.md");
for (const name of [...names, "studio_ui_request"])
  assert.ok(contract.includes(`\`${name}\``), `Tool index omits ${name}`);
for (const locale of locales) {
  const text = read(locale);
  assert.deepEqual(
    shape(text),
    baseline,
    `${locale}: locale structure, links or commands drifted`,
  );
  assert.ok(
    text.indexOf('id="try-it"') < text.indexOf('id="monster"') &&
      text.indexOf('id="monster"') < text.indexOf('id="install"'),
    `${locale}: show usage and monster before setup`,
  );
  assert.match(text, /26/);
  assert.match(text, /11/);
  for (const other of locales)
    assert.ok(
      targets(text).includes(other),
      `${locale}: language navigation omits ${other}`,
    );
}
const docs = readdirSync("docs", { recursive: true })
  .filter((name) => name.endsWith(".md"))
  .map((name) => path.join("docs", name));
for (const file of [
  ...locales,
  "CONTRIBUTING.md",
  "licenses/THIRD_PARTY.md",
  "packages/monster/README.md",
  ...readdirSync("packages/monster/skills", { recursive: true })
    .filter((name) => name.endsWith(".md"))
    .map((name) => path.join("packages/monster/skills", name)),
  ...docs,
])
  verifyLinks(file);
console.log(
  `Documentation OK: ${locales.length} locales, ${names.length} domain tools + app bridge, ${skillCount} skills; links/assets and locale parity verified.`,
);
