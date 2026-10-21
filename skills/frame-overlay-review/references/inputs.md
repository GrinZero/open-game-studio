# 输入与可重复性

## 准备帧

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

分别核实各视频的帧数和 FPS。变帧率、起始时间不同或速度变化的视频需要先建立明确的时间对应关系；单纯抽帧不会使它们同步。工具输入为已同步的图像序列，不重采样时间。

## 创建配置

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

## 分割

默认 `segmentation` 为 `{"mode":"border","threshold":45,"min_component":36,"border":20}`：计算每个视角边缘的中位 RGB，以 RGB 欧氏距离提取前景，剔除小连通域。边缘有角色或背景复杂时，应改用以下方式：

- 固定背景：`{"mode":"border","background":[16,20,28],"threshold":45,"min_component":36}`。
- 透明通道：`{"mode":"alpha","threshold":127,"min_component":36}`，输入必须实际包含 alpha。
- 显式蒙版：在对应 `sources.reference` / `sources.candidate` / `sources.baseline` 添加 `mask_directory` 和可选 `mask_pattern`；灰度值大于 `mask_threshold`（默认 127）为前景。蒙版必须与源图同尺寸。其他分割选项仍提供连通域过滤。

每个 source 还可用 `segmentation` 覆盖公共参数，适应参考与渲染背景不同；报告保存实际配置和每张图/蒙版的 SHA-256。保持修复前后使用相同分割规则，否则差值不能直接归因于修复。

报告包含 `report.json`、`per_frame.csv`、原坐标叠图、mask 对比图及无需服务器的 `index.html`。JSON 中 `summary` 是配对汇总，`worst_frames` 和 `regressed_frames` 方便 agent 定位。`coverage` 区分全部动作帧数、选择的帧数、实际有效帧数；局部预览始终标记 incomplete。
