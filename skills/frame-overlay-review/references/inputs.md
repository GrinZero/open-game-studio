# 输入、单帧多视角参数与可重复性

## 准备环境与抽帧

环境示例（环境放在用户缓存或任务目录，保持技能可移植）：

```bash
python3 -m venv /absolute/task/.venv
/absolute/task/.venv/bin/python -m pip install -r /absolute/skill/scripts/requirements.txt
```

视频抽帧用 `ffmpeg`，保持原画布和解码帧顺序，第一帧编号为 0：

```bash
mkdir -p /absolute/task/reference_frames
ffmpeg -i /absolute/reference.mp4 -map 0:v:0 -fps_mode passthrough -start_number 0 /absolute/task/reference_frames/frame_%04d.png
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames -of json /absolute/reference.mp4
```

分别核实各视频的帧数和 FPS。变帧率、起始时间不同或速度变化的视频需要先建立明确的时间对应关系；工具输入为已同步的图像序列或单帧图像，不重采样时间。

## 单帧多视角快速诊断与骨骼微调（`frame` 子命令）

`frame` 子命令专为 Agent-in-the-Loop 单帧迭代设计，无需先运行 `init` 生成 manifest：

- **图像输入**：
  - `--reference PATH` / `--candidate PATH` / `--baseline PATH`：既可传单帧合图文件，也可传帧目录（配合 `--frame N` 与 `--pattern 'frame_{frame:04d}.png'` 自动定位对应帧）。
  - `--ref-view VIEW=PATH` / `--cand-view VIEW=PATH` / `--base-view VIEW=PATH`：支持按视角分别传入单张图（如 `--ref-view front=/path/f.png --ref-view side=/path/s.png`），工具自动按视角拼接并记录固定视角框。
- **多视角切分与局部 ROI**：
  - `--layout {auto,single,2view,3view}`：默认 `auto`。若传入 `--rig-meta` 且其中含 `calibration.view_order`，优先按标定面板切分；否则根据宽高比自动切分为单视角（`front`）、双视角（`front`, `side`）或三视角（`left`, `front`, `right`）。
  - `--view NAME:left,top,right,bottom`：手动指定视角矩形（可重复）。
  - `--roi VIEW:NAME:left,top,right,bottom`：在指定视角内添加局部评分区域（如 `--roi front:right_arm:100,200,600,800`）。
- **3D 骨骼接入、单帧渲染与 FK 微调**：
  - `--rig-meta PATH`：传入 `studio_rig_meta.json`，启用 2D 骨骼正交投影叠加与误差簇骨骼归因（`bone_suggestions`）。
  - `--pose-json PATH`：传入 `predicted_trajectory.json` 或单帧姿态 JSON。
  - `--rig-skinning PATH` 或 `--blend PATH`：提供蒙皮权重 `.npz` 或角色 `.blend` 文件（首次从 `.blend` 自动提取并缓存蒙皮数据到临时目录）。
  - `--render-mode {auto,lbs,blender,none}`：`lbs` 使用内存向量化线性混合蒙皮（LBS）在约 15ms 内完成多视角渲染；`blender` 调用后台 Blender 渲染单帧；`none` 仅投射 2D 骨骼并计算归因建议，不重渲染图像。
  - `--tweak-view VIEW:BONE:deg`：在指定视角的 2D 成像平面内旋转骨骼（度），自动通过正运动学（FK）传递到所有子骨骼。
  - `--tweak BONE:drx,dry,drz` / `--set-euler BONE:rx,ry,rz` / `--tweak-root dx,dy,dz`：局部欧拉角增量、绝对角度或根骨骼位移微调。
  - `--auto-nudge-iters K`：自动根据当前帧外轮廓误差簇执行最多 `K` 轮阻尼骨骼微调并保留提升 IoU 的更新。
  - `--out-pose-json PATH`：将微调后的骨骼姿态写回 JSON，供下一轮迭代或最终烘焙使用。
  - `--overwrite`：允许在 Agent 循环中复用同一输出目录。

## 批处理配置（`init` + `review` 子命令）

```bash
python /absolute/skill/scripts/review.py init \
  --action attack --fps 24 --frame-count 124 \
  --reference /absolute/task/reference_frames \
  --baseline /absolute/task/before_frames \
  --candidate /absolute/task/after_frames \
  --pattern 'frame_{frame:04d}.png' \
  --view front:0,0,1280,1440 --view side:1280,0,2560,1440 \
  --output /absolute/task/manifest.json
```

manifest 内的相对路径以 manifest 所在目录为根；`init` 写绝对路径。矩形均为 `[left, top, right, bottom]`，右/下边界不包含；`views[].box` 位于整张输入图，`views[].regions[].box` 位于对应视角局部坐标。可添加固定 ROI，例如：

```json
{"name":"front","box":[0,0,1280,1440],"regions":[{"name":"shoulders","box":[200,200,1100,800]}]}
```

`alignment` 默认 `{"status":"unverified","evidence":""}`。独立核实相机与帧对应后，填写具体证据再将 status 改为 `verified`；此声明原样写入报告，不是工具自动校准的结论。

## 分割与输出诊断字段

默认 `segmentation` 为 `{"mode":"border","threshold":45,"min_component":36,"border":20}`：计算每个视角边缘的中位 RGB，以 RGB 欧氏距离提取前景，剔除小连通域。暗色棚拍渲染建议将 `--threshold` 设为 `12`–`20`；边缘有角色或背景复杂时，应改用以下方式：

- 固定背景：`{"mode":"border","background":[16,20,28],"threshold":16,"min_component":36}`。
- 透明通道：`{"mode":"alpha","threshold":127,"min_component":36}`，输入必须实际包含 alpha。
- 显式蒙版：在 manifest 的 `sources.*` 添加 `mask_directory`，或在 `frame` 子命令传 `--ref-mask` / `--cand-mask` / `--base-mask`。

报告包含 `panel_compare.png`（三合一多视角外轮廓诊断卡）、`report.json`、`per_frame.csv`、原坐标叠图、mask 对比图及无需服务器的 `index.html`。每个视角/区域的评估结果除 `iou`、`dice`、`boundary_mean_px`、`boundary_p95_px`、`boundary_within_tolerance` 外，还包含：
- `outline_passed`：当前视角外轮廓是否达到对齐阈值。
- `centroid_shift_px`、`bbox_delta_px`、`mean_contour_pull_px`：整体质心偏移、包围盒增量与轮廓平均拉动向量。
- `error_clusters`：按误差面积×最大距离排序的连通差异簇（含 `kind: protruding|missing`、`box`、`anchor_contour_px`、`target_contour_px`、`correction_vector_px` 及归因骨骼 `nearest_bone`）。
- `bone_suggestions`：汇总后的骨骼级旋转建议（含 `--tweak-view` 与 `--tweak` 可直接复用的参数字符串）。
