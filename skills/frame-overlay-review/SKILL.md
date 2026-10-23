---
name: frame-overlay-review
description: 同坐标逐帧比较参考视频与模型动画渲染，支持单帧多视角快速外轮廓对齐、方向性误差簇分析、骨骼归因与 Agent-in-the-Loop 动作精调，生成三合一诊断卡（panel_compare.png）、轮廓评分和修复前后配对报告。
---

# 同坐标逐帧叠图评分与单帧多视角外轮廓对齐

将同一动作、同一帧、同一相机视角的参考与渲染放在原始像素坐标下比较。支持两种工作模式：
1. **单帧多视角 Agent-in-the-Loop 动作精调（`frame` 子命令）**：无需预先创建 manifest，直接传入单帧合图或分视角图（自动识别 `single` / `2view` / `3view`），支持结合 `studio_rig_meta.json` 与姿态 JSON 执行 **FK 骨骼微调 + 快速 LBS 内存渲染（~15ms）/ Blender 渲染**，输出三合一外轮廓诊断卡 `panel_compare.png`、方向性误差簇（`error_clusters`）与下一轮骨骼旋转微调参数（`top_bone_suggestions`）。
2. **完整序列批处理验收（`init` + `review` 子命令）**：使用 manifest 对完整动作帧序列执行配对汇总、最差帧定位与防退步回归检查。

## 模式一：单帧多视角 Agent-in-the-Loop 轮廓对齐（`frame`）

1. **直接比较单帧多视角图像**（自动切分双视角 `front/side` 或三视角 `left/front/right`）：
   ```bash
   python <技能目录>/scripts/review.py frame \
     --reference /absolute/ref_0024.png \
     --candidate /absolute/cand_0024.png \
     --layout auto --threshold 16 --frame 24 \
     --output /absolute/review_f24 --overwrite
   ```
2. **结合 3D 骨骼元数据定位偏差骨骼并执行闭环微调**：
   传入 `--rig-meta` 与 `--pose-json`（以及 `--rig-skinning` 或 `--blend`），工具会自动将外轮廓误差簇归因到主干形变骨骼，并支持直接通过 `--tweak-view VIEW:BONE:deg`、`--tweak BONE:drx,dry,drz` 或 `--auto-nudge-iters K` 驱动 FK 骨骼旋转、重渲染当前帧并输出更新后的姿态文件 `--out-pose-json`：
   ```bash
   python <技能目录>/scripts/review.py frame \
     --reference /absolute/ref_0024.png \
     --rig-meta /absolute/studio_rig_meta.json \
     --pose-json /absolute/predicted_trajectory.json \
     --blend /absolute/rig.blend --render-mode lbs \
     --tweak-view front:LIMB_R_01_02:+8.5 \
     --auto-nudge-iters 2 \
     --out-pose-json /absolute/pose_f24_refined.json \
     --threshold 16 --frame 24 \
     --output /absolute/review_f24 --overwrite
   ```
3. **阅读单帧诊断输出**：
   - 终端 stdout 直接输出紧凑 JSON（含 `outline_passed`、`mean_iou`、`iou_gain`、`top_bone_suggestions` 和 `panel_compare` 路径）。
   - 查看 `panel_compare.png`（`VIDEO - FRAME N` | `OUR MODEL - RENDERED (含 2D 骨骼投影)` | `COMPARE + OUTLINE PASS/DIFF + 黄色修正拉动箭头`），根据 `top_bone_suggestions` 在循环中继续微调直至外轮廓吻合。

## 模式二：完整序列批处理（`init` + `review`）

1. 核实动作、帧编号、FPS、画布大小及各视角区域。记录相机投影、位置、朝向、尺度与时间对应关系的证据；证据不足时保留 `alignment.status: unverified`，仍可生成诊断报告。
2. 按 [输入与参数说明](references/inputs.md) 准备帧目录并执行 `scripts/review.py init` 创建 manifest。
3. 执行 `python <技能目录>/scripts/review.py review /absolute/manifest.json --output /absolute/new-review-dir`（调试可加 `--frames 0,24,48-60`）。
4. 阅读 `report.json` 与 `index.html` 的覆盖率、每个视角的配对分数、方向性误差簇与最差帧。

## 判断与后续修复

- 分数均在原分辨率计算；视角与局部 ROI 是各版本共用的固定矩形。工具拒绝尺寸不等的输入，不做平移、缩放、重心对齐或图像变形。
- `error_clusters` 区分 `protruding`（紫：当前模型多出轮廓）与 `missing`（青：参考轮廓未被覆盖），`correction_vector_px: [dx, dy]` 始终从当前模型轮廓锚点指向参考目标轮廓点。
- 结果描述为 **2D 轮廓匹配与方向性偏差诊断**；完整验收仍需结合全序列覆盖、坐标核实、全部目标视角和人工检查。

退出码：`0` 为完整、有效且已核实的比较；`2` 为覆盖不全、坐标未核实或有无效 mask 的诊断报告；`1` 为输入/执行错误。
