import path from "node:path";

// Python environments and cloud credentials belong to their runtime, not the plugin copy.
export function runtimeCopyFilter(entry, source, repo = process.cwd()) {
  if (entry !== "packages") return true;
  return !path
    .relative(path.join(repo, entry), source)
    .split(path.sep)
    .some(
      (part) =>
        [
          ".venv",
          "__pycache__",
          "node_modules",
          "var",
          "endpoint_registry.json",
        ].includes(part) ||
        part === ".env" ||
        (part.startsWith(".env.") && !part.endsWith(".example")),
    );
}
