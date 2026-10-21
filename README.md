<p align="center">
  <img src="docs/images/banner.svg" alt="Open Game Studio — Your local game production workbench for Codex" width="100%" />
</p>

<p align="center">
<a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a>
</p>

<h1 align="center">Open Game Studio</h1>

<p align="center"><strong>Your local game production workbench for Codex.</strong></p>

<p align="center">
  <img src="docs/images/badges.svg" alt="Alpha · Node.js 24+ · Codex MCP Apps · Local SQLite · MIT License" width="100%" />
</p>

Bring references, documents, media and 3D assets onto one visual canvas. Keep their versions, production decisions and exports connected to the project files on your machine.

> **Alpha · macOS installer · MIT License.** Local organization works on its own; AI production requires your own configured tools and accounts. README translations do not change the workbench UI, which currently uses Chinese labels.

[Try it](#try-it) · [Install](#install) · [Features](#features) · [Gallery](#gallery) · [Docs](#documentation) · [Limits](#limits)

---

<a id="try-it"></a>

## Try it in a conversation

Once installed, paste this into a new Codex chat:

```text
Open the Open Game Studio workbench.
Create a project named Forest Relics in a folder I choose.
Add a character brief and a reference image to a canvas,
then connect them as inputs to a character production group.
Show me the frozen input list before starting production.
```

Use the workbench to organize and inspect; use the conversation to tell Codex what to do. Connections supply inputs. Starting production is a separate action.

Want a starting point for file organization? Read the optional [project file guide (Chinese)](skills/studio-project/references/project-file-guide.md), then ask: `Adopt the standard Open Game Studio file guide in my game project.` You can choose individual rules; existing projects keep their layout unless you request a migration.

<a id="install"></a>

## Install and open

You need **macOS**, **Node.js 24+** (with npm), and a Codex CLI that provides `codex plugin`. The embedded workbench also needs a Codex host with MCP Apps support.

Run the one-line installer (or clone the repository manually):

```sh
curl -fsSL https://raw.githubusercontent.com/GrinZero/codex-game-studio/main/scripts/bootstrap.sh | bash
```

```sh
git clone https://github.com/GrinZero/codex-game-studio.git
cd codex-game-studio
npm ci
npm run install:local
```

The installer builds both workbenches, copies an independent runtime, installs `game-studio@game-studio-local`, and starts a user LaunchAgent for the browser workbench. This is a local source installation; no npm package or remote marketplace install is advertised.

1. Start a **new Codex chat** and say: `Open the Open Game Studio workbench.`
2. You can also select **Open Game Studio** from the sidebar’s **…** menu when your host exposes the registered entrypoint. If it is missing, restart Codex.
3. The browser fallback is [http://127.0.0.1:4310](http://127.0.0.1:4310). Run `npm run demo` from the checkout for the original **Star Maker** sample (shown as `星铸者 · 示例`).

The demo imports illustrations and a brief into two canvases. It runs no generation, review or engine validation.

For updates, removal, custom paths and direct MCP setup, see [Setup](docs/SETUP.md).

<a id="features"></a>

## What you can do

| Capability | What it gives you |
| --- | --- |
| Local projects | Create or link a folder, choose a cover, and keep existing files in place. |
| Versioned assets | Import file sets with SHA256 and immutable history. Canvas references stay pinned when newer versions arrive. |
| Visual canvases | Arrange assets and notes, connect inputs, reuse versions across canvases, and undo/redo layout changes. |
| Documents and media | Edit Markdown into new versions; preview images, audio, video and self-contained GLB models with animation controls. |
| Asset production | Draft image, video, model, animation or audio tasks, freeze references, and run them through local Codex with configured tools. |
| Character workflows | Follow brief → reference → candidate → appearance review → assembly → rig → deformation review → export, with saved decisions and partial reruns. |
| Traceable delivery | Export explicit versions with a file manifest and hashes. Keep engine acceptance as a separate check. |

Learn the interaction details in the [Workbench guide](docs/WORKBENCH.md).

<a id="gallery"></a>

## See the workbench

Real screenshots from an isolated local demo, using the repository’s original sample artwork. They show the current Chinese UI, not translated UI or completed production assets.

**Project library · dark theme**

![Project library · dark theme](docs/images/projects.png)

**Connected canvas · dark theme**

![Connected canvas · dark theme](docs/images/canvas.png)

[Image provenance](docs/images/README.md)

<a id="documentation"></a>

## Tools and documentation

**26 domain MCP tools + 1 app-only bridge · 11 workflow skills.** HTTP and embedded MCP Apps share the same project core and asset history.

| Start here | Reference |
| --- | --- |
| [Setup, data paths and MCP configuration](docs/SETUP.md) | Install, update, remove or connect directly. |
| [Workbench guide](docs/WORKBENCH.md) | Canvas editing, version selection and production handoffs. |
| [Tool contract](docs/TOOL_CONTRACT.md) | Complete tool index, input schemas and state rules. |
| [Verification record](docs/VERIFICATION.md) | What was tested and what still needs real-world validation. |
| [Implementation status](docs/IMPLEMENTATION.md) | Architecture, incomplete milestones and planned work. |
| [Monster module](packages/monster/README.md) | [Open in Colab](https://colab.research.google.com/github/GrinZero/codex-game-studio/blob/main/packages/monster/notebooks/colab.ipynb), Worker source and auto-discovered local client. |

<a id="limits"></a>

## Current limits

- **Early Alpha:** local projects, versioning and workflows are implemented. Built-in provider APIs, credential management, a complete cost panel, automatic game-directory scanning, database backup and Blender/Godot validation workers remain unfinished.

- **Bring your own production tools:** the local Codex executor does not bundle model services. Availability, login, fees and submission approval depend on each configured tool. A missing tool or required approval produces an explicit failure, not a placeholder asset. Stopping a local task does not cancel work already submitted remotely.

- **Preview is not acceptance:** GLB structure checks and file hashes do not establish mesh quality, editable Blend files, rig deformation or engine compatibility. Export manifests keep engine validation at `not_verified`.

- **Pinned means pinned:** new asset versions do not replace versions already used in canvases or reviews. SVG and glTF files with external dependencies are not supported for import. Undo/redo currently covers layout, not every graph operation.

- **Host support varies:** the MCP App entrypoint is registered and tested in a sandbox. Its visibility, caching and refresh behavior in the native Codex host depend on the host.

<a id="contributing"></a>

## Contributing

For development commands, checks, build markers and runtime synchronization, see [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports should include the displayed build marker, host, reproduction steps and expected behavior; remove credentials and private project files.

<a id="license"></a>

## License and sources

This project is licensed under the [MIT License](LICENSE). Dependency licenses and original sample provenance are recorded in [Third-party notices](licenses/THIRD_PARTY.md).

README presentation takes inspiration from [Tripo Studio for Codex](https://github.com/GrinZero/tripo-studio-plugin); the banner and demo screenshots here belong to this project.
