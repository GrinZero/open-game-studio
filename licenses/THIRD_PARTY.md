# Sources and dependencies

The optional [game project file guide](../skills/studio-project/references/project-file-guide.md) adapts organization and naming guidance from [Allar / Gamemakin](https://github.com/Allar/ue5-style-guide) and [bakneko](https://github.com/bakneko/godot-style-guide). Both use the MIT License. Their [Gamemakin notice](../skills/studio-project/references/third-party/allar-MIT.txt) and [Bakneko notice](../skills/studio-project/references/third-party/bakneko-MIT.txt) ship with the skill. [Source coverage](../skills/studio-project/references/guide-sources.md) distinguishes adopted rules, cross-engine adaptations and original additions.

The embedded workbench uses the official MCP Apps SDK (MIT), OpenAI MCP Extensions (Apache-2.0), and esbuild (MIT). Sidebar registration follows the documented SDK metadata contract. No implementation code from the user's other workbench plugins is bundled.

The initial domain model, UI, workflow and skill text are original implementations guided by the supplied product design. This build does not include ohmygame, Charakuru or UniMate source, skill text, model weights or character materials.

React and React DOM (MIT), React Flow (MIT), lucide-react (ISC), Express (MIT), multer (MIT), Zod (MIT), the MCP TypeScript SDK (MIT), model-viewer (Apache-2.0), Vite (MIT), TypeScript (Apache-2.0), tsx (MIT) and Playwright (Apache-2.0) retain their own notices in installed packages. Release packaging must include a generated transitive dependency notice inventory.

The PNG illustrations and their SVG source in samples are original demonstration materials covered by the project's [MIT License](../LICENSE). No production model or quality evidence is fabricated from them. Original workbench code, documentation and skills use that license except Monster and the exported `monster-workflow` skill, which use [AGPL-3.0-only](../packages/monster/LICENSE). Third-party dependencies retain their own licenses and notices.

## Monster migration sources

Monster also ships its own [AGPL-3.0-only license](../packages/monster/LICENSE) , [licensing scope](../packages/monster/LICENSING.md) and standalone [source and dependency notices](../packages/monster/THIRD_PARTY.md); all three are included in its deployment archives.

The `packages/monster` module was extracted from the user-supplied [MonsterStudio Colab Notebook](https://drive.google.com/file/d/1Zs1rJSnQC9fa7G76Qdj6ALmArBuom179/view). The notebook author has confirmed ownership of the original pipeline and algorithm code. The migration preserves the original algorithm sources and upstream references; it does not relicense third-party dependencies or model weights.

The Colab setup downloads [VAST-AI SkinTokens](https://github.com/VAST-AI-Research/SkinTokens), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [ComfyUI](https://github.com/Comfy-Org/ComfyUI), [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) and [MiniMax H3 weights](https://huggingface.co/Comfy-Org/MiniMax-H3) at the revisions retained in the installation scripts. Training also loads Meta DINOv2. The version-specific core dependency inventory, including the H3 upscaler, is maintained in [Monster third-party notices](../packages/monster/THIRD_PARTY.md). These projects, Blender/bpy, Gradio and their dependencies remain subject to their respective upstream notices and model terms. No model weights or upstream repository checkout are bundled in this repository or its Monster deployment archives.
