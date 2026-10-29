# Licensing and external dependencies

The skill instructions and scripts are MIT licensed; see `../LICENSE`. No website implementation, proprietary model, website download, account data or authentication material is distributed with this skill.

`package-lock.json` pins the Node dependencies. Their original license notices remain in the installed packages:

| Dependency | Version | Upstream license |
| --- | --- | --- |
| @gltf-transform/core, extensions, functions | 4.5.1 | MIT |
| draco3dgltf | 1.5.9 | Apache-2.0 |
| meshoptimizer | 1.3.0 | MIT |
| sharp | 0.35.5 | Apache-2.0; bundled native dependencies have their own notices |

Optional KTX-Software is installed separately from the official Khronos distribution, with its included license notices (Apache-2.0 and third-party components). Tested with 4.4.2. Its binaries are not included in the portable source archive. If distributing an environment with node_modules or native binaries, preserve all upstream notices rather than only this summary.
