import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

// Packages own their skills; the plugin's root skills directory is the install surface.
export function exportPackageSkills(repo = process.cwd()) {
  const packages = path.join(repo, "packages");
  const exported = [];
  if (!existsSync(packages)) return exported;
  for (const pkg of readdirSync(packages, { withFileTypes: true })) {
    if (!pkg.isDirectory()) continue;
    const sourceRoot = path.join(packages, pkg.name, "skills");
    if (!existsSync(sourceRoot)) continue;
    for (const skill of readdirSync(sourceRoot, { withFileTypes: true })) {
      if (!skill.isDirectory()) continue;
      const source = path.join(sourceRoot, skill.name);
      if (!existsSync(path.join(source, "SKILL.md"))) continue;
      if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(skill.name))
        throw new Error(`Invalid package skill name: ${skill.name}`);
      const target = path.join(repo, "skills", skill.name);
      const relative = path.relative(repo, source).split(path.sep).join("/");
      const marker = path.join(target, ".package-skill.json");
      if (existsSync(target)) {
        if (!existsSync(marker) || JSON.parse(readFileSync(marker, "utf8")).source !== relative)
          throw new Error(`Package skill conflicts with an existing skill: ${skill.name}`);
        rmSync(target, { recursive: true });
      }
      mkdirSync(path.dirname(target), { recursive: true });
      cpSync(source, target, {
        recursive: true,
        filter: (srcPath) => {
          const base = path.basename(srcPath);
          return base !== "__pycache__" && !base.endsWith(".pyc");
        },
      });
      writeFileSync(marker, JSON.stringify({ source: relative }, null, 2) + "\n");
      exported.push({ name: skill.name, source: relative });
    }
  }
  return exported;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href)
  console.log(JSON.stringify({ packageSkills: exportPackageSkills() }));
