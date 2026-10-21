# Monster sources and third-party dependencies

Copyright (c) 2026 GrinZero. Monster's project-owned code, documentation and skills use **AGPL-3.0-only** from version 0.2.0. See [LICENSE](LICENSE) and [licensing scope and obligations](LICENSING.md). Third-party code, dependencies, model weights and external assets retain their own licenses. The AGPL does not replace them or authorize restricted model use.

## Source provenance

The original author confirmed ownership of the self-written pipeline and algorithm code in the supplied [MonsterStudio Notebook](https://drive.google.com/file/d/1Zs1rJSnQC9fa7G76Qdj6ALmArBuom179/view) and optimized [Colab Notebook](https://colab.research.google.com/drive/1hmk2iXWntDqXbujEE1i3_9hCcsPLvcuR). This confirmation does not extend to their third-party dependencies. Previously granted MIT permissions remain available for earlier copies; the [prior MIT notice](licenses/Monster-prior-MIT.txt) is retained.

## Core upstream inventory

Checked on 2026-10-22 (Asia/Taipei), against the revisions selected by `scripts/bootstrap-skintokens.sh` and `scripts/setup-comfy.py`, or the floating upstream noted below. This inventory records upstream declarations, not a claim that every transitive dependency or deployment is cleared for commercial use.

| Component | Selected revision or version | Upstream declaration and evidence | Integration and boundary |
| --- | --- | --- | --- |
| SkinTokens code | `273b691d35989d71cd17ff2895fdc735097b92d1` | [MIT](https://github.com/VAST-AI-Research/SkinTokens/blob/273b691d35989d71cd17ff2895fdc735097b92d1/LICENSE), copyright 2025 VAST-AI-Research; [retained text](licenses/SkinTokens-MIT.txt) | Downloaded separately; bootstrap patches two upstream bpy files. Keep notices and modification details when redistributing. |
| SkinTokens FSQ-CVAE / TokenRig weights | `VAST-AI/SkinTokens`, floating download; observed model-card revision `79736cad0fd84de384d5eede659b4ebd24effe33` | [Model card declares MIT](https://huggingface.co/VAST-AI/SkinTokens/blob/79736cad0fd84de384d5eede659b4ebd24effe33/README.md); no separate LICENSE in the observed model repo | `download.py --model` fetches two checkpoints. Model-card metadata is recorded separately from the code license; training datasets and input assets are not licensed by that declaration. |
| Qwen3-0.6B configuration | Floating upstream from SkinTokens `download.py` | [Apache-2.0](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/LICENSE) | SkinTokens downloads configuration/tokenizer files, ignoring `.bin` and `.safetensors`. Preserve applicable notices if redistributing. |
| FlashAttention | `965c9212e6be33c821ee9f2b461564c5c91a2503` | [BSD-3-Clause](https://github.com/Dao-AILab/flash-attention/blob/965c9212e6be33c821ee9f2b461564c5c91a2503/LICENSE); [retained text](licenses/FlashAttention-BSD-3-Clause.txt) | Compiled in the external SkinTokens environment; vendored/submodule code can have additional notices. |
| Blender / bpy | `bpy==4.2.0` | [GPL-2.0-or-later source declaration](https://github.com/blender/blender/blob/v4.2.0/source/blender/python/intern/bpy.cc), [Blender FAQ on Python API scripts](https://www.blender.org/support/faq/) | Monster directly uses bpy/mathutils. Consider GPL v3 and the GPL/AGPL section 13 combination provisions; Blender remains GPL, not relicensed by Monster. |
| ComfyUI | `7f7fd918ed0815c5d5ef2ccfe44bfc8e72cfd217` | [GNU GPL v3 text](https://github.com/Comfy-Org/ComfyUI/blob/7f7fd918ed0815c5d5ef2ccfe44bfc8e72cfd217/LICENSE); [retained text](licenses/GPL-3.0.txt) | Separate checkout/server called over loopback HTTP. Its own code and bundled dependencies retain their licenses. |
| ComfyUI-VideoHelperSuite | `4d907bee61e92c2e65af3bd6383a4e4d356126d1` | [GNU GPL v3 text](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/4d907bee61e92c2e65af3bd6383a4e4d356126d1/LICENSE) | Installed as a ComfyUI custom node; not an MIT component. |
| H3 latent-upscaler custom node | `40316cf008b2fd8663263270669eb4da23f89d2c` | [MIT](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/40316cf008b2fd8663263270669eb4da23f89d2c/LICENSE), copyright 2026 LBH-123-AI; [retained text](licenses/H3-Upscaler-node-MIT.txt) | Separate code checkout. Its MIT declaration does not replace ComfyUI's GPL or the weights' terms. |
| H3 latent-upscaler weights | `3f941d5d182014dd5c0a5e16330420ee2d4aa0c6`, BF16 v1 checkpoint | [Model card declares Apache-2.0](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/3f941d5d182014dd5c0a5e16330420ee2d4aa0c6/README.md); no separate LICENSE in the checked path | Downloaded separately for 2K processing; H3 used alongside it still has H3 terms. |
| DINOv2 standard backbone | Floating `torch.hub.load("facebookresearch/dinov2", ...)` | [Standard DINOv2 code and weights: Apache-2.0](https://github.com/facebookresearch/dinov2#license); [retained Apache text](licenses/Apache-2.0.txt) | The standard backbone is distinct from Cell-DINO / XRay-DINO variants with noncommercial terms. Current loader does not pin a commit, so recheck the actual cached revision when deploying. |
| Gradio / gradio-client | `gradio==5.49.1`; client `>=1.4.0,<2` | [Apache-2.0](https://github.com/gradio-app/gradio/blob/gradio%405.49.1/LICENSE) | Python API and client; range-selected client releases must be checked at the installed version. |
| MiniMax H3 repackaged weights | `Comfy-Org/MiniMax-H3@e5eb578a89295337b8ff433a035929ce0279e0b6` | [Pinned model card](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/README.md) points to the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE); [retained snapshot](licenses/MiniMax-H3-Community.txt) | Downloaded into a separate ComfyUI model cache. Custom model terms, not MIT/Apache/AGPL. The Qwen3-VL encoder has additional Apache-2.0 provenance noted in H3's license. |

## MiniMax H3 terms requiring separate attention

The checked Comfy-Org model revision links to the original license through a floating `main` URL. The retained snapshot is the original MiniMax repository's license at `42ed227ee7df40d41602854ae760620d6eb651fe`, observed on the audit date (license date 2026-08-02). It records what was reviewed; it does not establish which terms applied to every earlier download. Verify the terms for your acquisition and deployment.

The snapshot contains, among other requirements:

- The territory grant excludes the European Union, United Kingdom, Republic of Korea and United States of America. Deployment in those territories requires an appropriate separate grant; this is not an unrestricted worldwide model license.
- Commercial products/services generating more than USD 20 million in annual revenue require prior written authorization from MiniMax. Commercial interfaces using the model must prominently display its name.
- Recipients/users of relevant products or services must receive the agreement, and use restrictions / Acceptable Use Policy must be passed through as specified. Redistribution of model works can require a NOTICE and modification notices.
- The licensor claims no rights over generated outputs, but users remain responsible for their use; outputs do not erase third-party asset rights or the model's other obligations.

These are a summary, not additional restrictions attached to Monster's AGPL. The [complete agreement and Acceptable Use Policy](licenses/MiniMax-H3-Community.txt) govern the model grant.

## Packaging and unresolved scope

Monster's deployment archive contains its own source and these notice/license texts. It does not contain upstream repository checkouts, model weights, user projects, runtime credentials, datasets or caches. The installer downloads dependencies later; this does not remove obligations for components actually used, modified, combined or redistributed.

The bootstrap modifies SkinTokens' bpy server bind address and vertex-group index cleanup. When shipping that patched checkout, preserve its MIT notice and identify these changes. The retained license texts are not a complete source offer for any third-party binary distribution.

This check is limited to the core components above. It does not enumerate all dependencies installed by upstream requirements, CUDA/NVIDIA packages, FFmpeg codecs/build options, or licenses of input meshes, videos and training data. Some dependencies and models float rather than use immutable revisions. A commercial deployment or a bundle including those components needs an inventory of its actual installed versions and corresponding notices before claiming complete clearance.
