import { readFileSync } from "node:fs";
import { StudioCore } from "../src/server/core.js";

// Destinations and a consistent recovery backup must be prepared before applying.
// This command changes references only; it never deletes original files.
const [planFile, apply] = process.argv.slice(2);
if (!planFile || apply !== "--apply") {
  throw new Error(
    "Usage: tsx scripts/relocate-asset-files.ts PLAN.json --apply (prepare backup and archive files first)",
  );
}
const plan = JSON.parse(readFileSync(planFile, "utf8"));
if (typeof plan.projectId !== "string" || !Array.isArray(plan.records))
  throw new Error("Plan requires projectId and records");
const core = new StudioCore();
try {
  console.log(
    JSON.stringify(
      core.relocateAssetFiles(plan.projectId, plan.records),
      null,
      2,
    ),
  );
} finally {
  core.close();
}
