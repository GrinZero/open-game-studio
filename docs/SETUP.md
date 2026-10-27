# Setup

[README](../README.md) · [Workbench guide](WORKBENCH.md) · [Contributing](../CONTRIBUTING.md)

## Local installation on macOS

Requirements: macOS, Node.js 24+ with npm, and a Codex CLI with `codex plugin` support. Embedded UI additionally requires an MCP Apps-capable host.

One-line install (clones or updates the source under `~/.local/share/codex-integrations/game-studio-local/source` and runs the local installer):

```sh
curl -fsSL https://raw.githubusercontent.com/GrinZero/codex-game-studio/main/scripts/bootstrap.sh | bash
```

Or clone the repository manually:

```sh
git clone https://github.com/GrinZero/codex-game-studio.git
cd codex-game-studio
npm ci
npm run install:local
```

The installer builds HTTP and MCP Apps UI, copies the runtime and dependencies outside the plugin cache, installs and enables `game-studio@game-studio-local` through a local marketplace, and starts the browser service with a user LaunchAgent. It currently supports macOS only. There is no published npm installation path in this repository.

Start a new Codex chat and ask `Open the Open Game Studio workbench.` The sidebar's **…** menu may also expose **Open Game Studio**. If the host has not discovered the entrypoint, restart Codex. The browser fallback is http://127.0.0.1:4310.

From the repository, `npm run demo` registers the original `星铸者 · 示例` project under `var/demo/star-maker`. Re-running it does not recreate an already registered demo. It imports original sample illustrations and documents, but performs no generation, review or engine validation.

## Updates and removal

Pull the repository updates and re-run `npm run install:local` to replace the owned runtime and refresh the plugin installation. The installer retains the project registry and project directories, which live outside that runtime. Maintainers updating an existing runtime without reinstalling should use the [synchronization procedure](../CONTRIBUTING.md#build-and-synchronize-an-installed-runtime).

Remove the plugin:

```sh
codex plugin remove game-studio@game-studio-local
```

The browser LaunchAgent is independent. Stop it before removing its login-startup configuration if you no longer need the browser service:

```sh
launchctl bootout gui/$(id -u)/local.codex.game-studio
```

Its plist is `~/Library/LaunchAgents/local.codex.game-studio.plist`. Removing that file disables future login startup. To start an existing configuration again:

```sh
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.codex.game-studio.plist
```

Plugin removal does not delete project files, asset revisions or the registry.

Character production data follows the [character workspace v1 contract](../packages/monster/skills/monster-workflow/references/character-workspace.md): one stable character directory under the user's project, with versioned inputs, training and actions. Existing SOP paths and review ledgers remain in place until the documented path-adapter migration is implemented.

## Paths and configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `STUDIO_HOME` | `~/.game-studio` | Registry directory; use the same absolute path for HTTP and MCP |
| `PORT` | `4310` | Browser service and signed media URL port; HTTP and MCP must agree |
| `STUDIO_CODEX_BIN` | `/opt/homebrew/bin/codex` | Local executor for asset production; configure in the actual HTTP/MCP service environment |
| Runtime | `~/.local/share/codex-integrations/game-studio-local/runtime` | Installed code and dependencies, outside the plugin cache |
| Registry | `~/.game-studio/registry.db` | Registered project locations |
| Project metadata | `<project>/.game-studio/studio.db` | Authoritative project database |
| Asset revisions | `<project>/.game-studio/assets/` | Immutable imported file sets |
| Production outputs | `<project>/.game-studio/generated/` | Task files, logs and receipts |
| Exports | `<project>/.game-studio/exports/` | Version-specific delivery folders and manifests |

For example, set `STUDIO_HOME` and `PORT` when invoking the installer to configure both services:

```sh
STUDIO_HOME=/absolute/path/studio-registry PORT=4315 npm run install:local
```

The product name is **Open Game Studio**. Existing identifiers remain `game-studio`, `codex-game-studio`, `ui://game-studio/workbench` and `.game-studio` for installation and data compatibility.

HTTP listens on 127.0.0.1 and checks Host and Origin. MCP Apps read project data through the app-only bridge, with signed binary HTTP media reads and MCP file chunks as a fallback. The embedded UI does not require unrestricted localhost API access.

## Run without installing the plugin

```sh
npm ci
npm run build
npm start
```

Open http://127.0.0.1:4310. For development, use `npm run dev` instead of `npm start`. The core is not macOS-specific; the local installer and native folder picker are. On other systems, enter project paths manually. Non-macOS installation has not been verified.

## Direct stdio MCP setup

After `npm ci` and `npm run build`, the repository can serve stdio MCP with `npm run mcp`. It writes no runtime logs to stdout. Alternatively, configure absolute paths in Codex:

```toml
[mcp_servers.game-studio]
command = "node"
args = ["/absolute/path/codex-game-studio/node_modules/tsx/dist/cli.mjs", "/absolute/path/codex-game-studio/src/server/mcp.ts"]

[mcp_servers.game-studio.env]
STUDIO_HOME = "/absolute/path/studio-registry"
PORT = "4310"
```

Ensure Codex can find Node.js 24+. Run the HTTP service separately with the same registry and port for the browser fallback and fast binary media route. The installed plugin config is generated at `~/.local/share/codex-integrations/game-studio-local/plugins/game-studio/.mcp.json`; inspect the active configuration before changing service paths.

Detailed inputs and lifecycle rules are in the [tool contract](TOOL_CONTRACT.md). Host entrypoint registration does not guarantee every host exposes or refreshes it; verification scope is recorded in [VERIFICATION.md](VERIFICATION.md).
