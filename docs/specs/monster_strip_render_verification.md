# EEVEE 动画条带渲染验收（2026-10-27）

实现入口为 `packages/monster/worker/strip_renderer.py`，由 Stage 2 的正式 5,000 样本生成流程调用。旧 `video2rig_blender.StripRenderer` 导入继续兼容。默认 EEVEE 8 spp，Cycles GPU/OIDN 作为显式选项保留。Colab 固定 bpy 4.5.3；其 Python 3.11 Linux wheel 与已固定 NumPy 1.26.4 兼容。EEVEE 灯光和阴影链接从 Blender 4.3 起支持（[官方发布说明](https://developer.blender.org/docs/release_notes/4.3/eevee/)），旧 4.2 条带执行会明确报错并提示升级 Cell 2 或关闭条带。

每个槽位保存独立骨架、Action、变形器、道具父级及灯光状态；不可变网格共享以减少内存。副本沿当前相机水平轴排列，奇偶前后错开 2.5m，保留半面板间隔。完整批次和尾批均按原面板标定裁切，支持横排双/三视图和上下堆叠。相机、输出格式、模糊参数、源对象显隐和临时副本在成功或异常后恢复。清晰/模糊批次隔离，条带快门固定 0.5，标签为中心帧 1。

目标批量为 16；默认条带最大宽度 16384px，896px 面板含间隔时实际最多 12 份。双视图每批仍须两次渲染；不能把 16 副本当作 16 倍吞吐。`keypoints_gt.json.render_pipeline` 保存实际批量、批次数、调用次数和渲染耗时。引擎、采样数、批量策略和渲染契约纳入数据集签名，使旧缓存失效。

v3 保留 896px 原图，在 ReaderDataset 中用完整变形网格（含道具、长尾、翅尖）加 12% 余量生成正方形窗口。RGB、关节点、上下文与稠密场共享映射后送入 448px Reader；旧数据缺少网格框时继续使用原逻辑。裁剪收益依赖实际占画幅比例，无法恢复已经离开原图画幅的表面。

## 实际渲染与回归

- `fixture_strip_render.py`：Blender 5.3 Beta / EEVEE / 8 spp，小型单骨蒙皮资产，96px 双视图，预热后比较 17 个相同姿态；34 次单张渲染对比 4 次条带渲染，独立开关清晰/运动模糊，另验证完整 16 副本 + 1 尾批、堆叠输出、宽度自动降批及注入异常后的场景恢复。
- 记录于 `var/qa/strip-render/actual/report.json`：一次本机测量逐张约 2.24s、条带约 0.41s，约 5.4×；平均绝对像素误差逐样本约 0.52–0.55 / 255（含 JPEG 编码）。这不是复杂角色、2K、Colab G4 或完整 5,000 张采样/可见度/稠密场/训练全流程的速度结论。
- `test_strip_renderer.py`：包围框保留极端顶点、448px 裁剪和坐标逆映射、参数改变触发缓存失效、旧 EEVEE 版本阻止、真实渲染调用与异常恢复。
- `MONSTER_TEST_BLENDER=... npm run monster:check` 最终 145 项全部通过、零跳过，覆盖真实 Blender 标签 → CPU 训练 → checkpoint 重载 → GLB 回读，以及 P1–P4 形式化门禁。日志保存在 `var/qa/strip-render/monster-check.log`。
- `npm test`：73 项全部通过；`npm run build` 和 `npm run check:distribution` 通过。前端/服务测试日志保存在同目录。
- `npm run check:docs` 仍被已有 `docs/video2rig_vs_monster_workflow_deep_dive.md` 的 `file://` 本地引用阻断：目标文件实际存在，但检查器将 URI 当相对文件路径；此次未扩展修复该检查器。

## 部署边界

同步本地安装 runtime 及 HTTP/MCP 界面产物，再刷新用户原有 Chrome 工作台页面并核对版本。Python worker 是逐次启动的进程，下一次执行会从新运行目录加载；此次没有启动或中断任何角色的训练任务，也没有改变角色审核状态。Colab 的 GPU 环境、bpy 依赖和进程未在此次本地测试中更新；须无活动任务时运行新 Cell 1 → Cell 2 → Cell 4，随后在真实角色上验收画质与吞吐。
