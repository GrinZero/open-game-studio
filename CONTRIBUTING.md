# Contributing to Open Game Studio

For installation and everyday use, start with the [English README](README.md) or [Setup](docs/SETUP.md). This guide covers development and maintenance. The workbench uses the [MIT License](LICENSE); Monster and its exported workflow skill use [AGPL-3.0-only](packages/monster/LICENSE), with scope described in [Monster licensing](packages/monster/LICENSING.md). Contributions are provided under the license applicable to the component; preserve the applicable notices for third-party code or assets.

## Develop locally

Use Node.js 24+; storage relies on Node's built-in SQLite.

```sh
npm ci
npm run dev
```

The development workbench listens on http://127.0.0.1:4310. `npm start` serves the production build after `npm run build`. Use an isolated absolute `STUDIO_HOME` and a different `PORT` for test instances so they do not share the user's registry.

## Required checks

```sh
npm run check:docs
npm run check
npm test
npm run build
npm run check:distribution
npm run monster:check
npm run monster:package
```

Documentation checks validate local links and assets, locale parity, installation examples, and tool/skill counts against the implementation. They compare requirements rather than exact translated headings. The documentation workflow runs this check on pushes and pull requests.

For changes to browser behavior, start an isolated service and run the real browser smoke test:

```sh
STUDIO_HOME=/absolute/path/browser-test-registry PORT=4311 npm run dev
# In another terminal; install Chromium first if it is unavailable.
npx playwright install chromium
npm run test:ui
```

The browser smoke test defaults to port 4311; `STUDIO_TEST_URL` can select another instance. It creates test projects and saves diagnostic screenshots in `var/qa`. For the opaque-origin MCP Apps sandbox, run `npm run test:mcp-ui`; this script manages its own temporary registry and services. These checks do not submit paid model operations or prove production asset quality. See [Verification](docs/VERIFICATION.md) for the scope of past checks.

## Build and synchronize an installed runtime

Follow [AGENTS.md](AGENTS.md). Every plugin update must receive a fresh build marker and be synchronized to the active installation. Run `npm run build` once after the final edits; documentation-only work does not need repeated builds of unchanged code.

The visible marker is `v<package.json version>-YYYYMMDDHHmmss`, generated during the build in `Asia/Taipei`. It is never generated when opening a page. HTTP and MCP Apps must contain the same marker, verified by `npm run check:distribution`. This UI marker is independent of the server compatibility constant `studioBuild`; a presentation change alone does not require changing that constant.

Before synchronization, inspect the installed MCP configuration and the LaunchAgent to confirm their runtime, `STUDIO_HOME` and `PORT`. The default runtime is `~/.local/share/codex-integrations/game-studio-local/runtime`.

```sh
node scripts/sync-runtime.mjs
```

The synchronization script checks the configured runtime, backs up the owned files and copies the build, source and documentation. It does not relocate project data. Prefer this path to reinstalling the whole plugin just to refresh documentation or UI. If server code changes, load it through the smallest necessary service restart; existing stdio connections may need reconnection. For versioned releases, align `package.json`, `package-lock.json`, `plugin.json` and the MCP server version before using the local installer.

Verify served HTTP and MCP artifacts, then refresh the user's actual workbench and inspect its visible marker. A newly opened test page or a chat-side panel does not prove an already open main window has refreshed. If the actual window is inaccessible, record the limitation and do not claim complete hot update.

## Repository map

| Path                | Responsibility                                                                          |
| ------------------- | --------------------------------------------------------------------------------------- |
| `src/server/`       | Shared core, HTTP, stdio MCP and production execution                                   |
| `src/shared/`       | Domain types, input rules and build marker                                              |
| `src/ui/`           | React, React Flow and lazy-loaded GLB preview                                           |
| `packages/monster/` | Monster Python Worker, client integration, protocol, Colab entry and deployment tooling |
| `skills/`           | Setup and production workflow skills                                                    |
| `samples/`          | Original illustrations, brief and viewer sample                                         |
| `templates/`        | Character workflow template                                                             |
| `tests/`            | Domain, media and real stdio MCP checks                                                 |
| `scripts/`          | Installer, build, runtime sync, demo and browser checks                                 |
| `docs/`             | Setup, workbench guide, contracts and verification                                      |

The registry stores project locations. Each project's `.game-studio/studio.db` is authoritative; asset revisions live in `.game-studio/assets/<assetId>/<revisionId>/`, production outputs in `.game-studio/generated/`, and exports in `.game-studio/exports/`. JSON files are snapshots or exports, not a second source of truth.

## Monster module

The repository is an npm workspace monorepo. The current workbench retains its root `src/` layout; the mixed Python/Node Monster module lives under `packages/monster/`. Keep its Worker, contracts, client and Colab deployment entry together. See the [Monster guide](packages/monster/README.md) for CPU-only checks, deployment bundles, local client calls and GPU verification boundaries. Both local installation and runtime synchronization copy `packages/`; neither deploys to an already running Colab instance.

## Bundled project file guide

The optional [project file guide](skills/studio-project/references/project-file-guide.md) is user-facing guidance for game projects, not a mandatory layout for this plugin repository. Keep its canonical copy, naming reference, engine profiles, VCS profiles, source coverage and upstream notices under `studio-project/references/` so installation and runtime synchronization carry the complete guide with the skill. The project skill only applies selected rules after user opt-in; registration must keep existing file organization and VCS choices. Keep the quick start approachable and the reference complete, preserve engine-specific requirements, and distinguish conversational assistance from implemented automation. The VCS reference is tool-neutral project guidance, not implemented service connectors. Validate relative links across the entire bundled guide when changing it.

## Keep documentation honest

Maintain English in `README.md`, with Simplified Chinese, Traditional Chinese, Japanese and Korean counterparts. Keep section order, stable navigation anchors, table dimensions, commands, links and images aligned. Use natural language for each locale rather than sentence-by-sentence translation. The translated READMEs describe the same product; they do not imply a localized UI.

Keep maintainer procedures here and detailed interactions in [Workbench](docs/WORKBENCH.md). Verify capability claims against the code, not the previous README. Preserve version pinning, saved review history and user data. Do not label previews, synthetic test fixtures, structural validation or file hashes as engine acceptance.

README artwork and screenshots must live in the repository. See [image provenance](docs/images/README.md) before replacing gallery assets. Third-party notices belong in [licenses/THIRD_PARTY.md](licenses/THIRD_PARTY.md).
