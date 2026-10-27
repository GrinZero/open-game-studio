# Blender / bpy 最新稳定版升级（2026-10-27）

[Blender 官方发布记录](https://www.blender.org/releases/) 与 [bpy PyPI](https://pypi.org/project/bpy/) 确认：最新稳定版为 5.2.3 LTS（2026-10-22），5.3 为 Beta。

## 已安装版本与依赖边界

- 桌面默认 `/Applications/Blender.app` 已安装官方 Blender 5.2.3 LTS，`~/.local/bin/blender` 原有入口继续指向它。安装包 SHA-256 为 `acc05374485a272342d4b323c15ac01953e7f697da2f1bad8af9c3bf087267a4`，与官方校验文件一致。原 5.3 Beta 保存在 `/Applications/Blender 5.3 Beta.app`；原 5.1.2 继续保留，用户配置与工程文件未迁移或覆盖。
- 本机独立 bpy 环境为 `~/.local/share/codex-integrations/game-studio-local/blender-python-5.2.3/`：Python 3.13.12、bpy 5.2.3、NumPy 2.5.4、SciPy 1.18.1、Pillow 13.0.0、scikit-image 0.26.0、OpenCV headless 5.0.0.93、threadpoolctl 3.7.0。安装时均核对了 PyPI 最新版本；传递依赖由解析器选择，完整记录位于 `var/qa/blender-upgrade/installed-packages.txt`。
- bpy 5.2.3 官方 wheel 要求 Python 3.13 / NumPy ≥2.2；SkinTokens 依赖 Python 3.11 / NumPy 1.26.4，故其内部 bpy 使用该 ABI 的最新 LTS 4.5.15。正式摄影棚的预检、姿态渲染、考试和 GLB 烘焙调用均改为独立 bpy 5.2.3 环境；SkinTokens 推理保持独立。

## 部署实现

`contracts/blender-runtime.json` 定义版本与 ABI；`worker/blender-requirements.txt` 固定渲染依赖；`scripts/bootstrap-blender.sh` 在 Colab 通过 uv 创建 `/content/monster-bpy/envs/5.2.3`，校验安装后更新 `venv` 入口，不覆盖已有非符号链接目录。

Cell 2 会核对 SkinTokens 的 `ready.json.bpy`，旧版本或旧标记会触发升级；独立渲染环境每次均执行依赖核验，不再因已有 ready 文件而跳过。更新前沿用现有 Worker 忙碌守卫，不中断活动 GPU 任务。`blender_runtime.resolve_bpy_python()` 供服务与流水线统一解析；桌面 SOP 优先发现同一环境。渲染目标版本纳入数据集签名，实际 Blender 版本写入渲染标定。

## 验证证据与边界

- 官方桌面程序 `--version` 实测为 5.2.3 LTS，终端入口同步生效。
- 使用新 bpy / 新直接依赖实际运行 `fixture_strip_render.py`：16 副本、尾批、双视图、堆叠、运动模糊、宽度拆批和故障恢复通过。原始日志与收据在 `var/qa/blender-upgrade/bpy-render/`；小型 17 姿态测试约 5.6×，不作为真实资产/G4 吞吐承诺。
- 使用新 bpy / 新依赖实际运行 `fixture_v3_blender.py`：相机投影、完整表面权重回退、稠密标签、完整条带、独立道具 6DoF、分数帧 GLB 烘焙/回读及 FPS 时间标定通过。
- 在本地独立临时目录实际运行 `bootstrap-blender.sh`，安装固定依赖、校验 Blender 5.2.3 / Python 3.13 / NumPy 2.5.4 并建立 `venv` 入口成功；日志 `var/qa/blender-upgrade/bootstrap.log`。
- `test_blender_runtime.py` 覆盖 Python ABI 隔离、显式路径优先、失效路径报错、旧 ready 标记升级、新 renderer 必须准备。
- 完整 Monster 检查：153 项通过、零跳过；插件测试：73 项通过。构建、HTTP/MCP 同版本及本地 runtime 验证日志位于 `var/qa/blender-upgrade/`。

`colab sessions` 返回当前账号无活动会话（日志 `var/qa/blender-upgrade/colab-sessions.log`）。本次更新本机软件、本地插件运行目录及 Colab 部署脚本；没有启动或修改云端 GPU 实例。现存 Colab 需要空闲时运行新版 Cell 1 → Cell 2 → Cell 4，才能实际安装新依赖并让云端进程生效。
