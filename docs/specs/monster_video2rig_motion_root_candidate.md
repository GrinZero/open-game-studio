# 空组织链与起始姿态修复（已合入0.3.9）

此修复补充原Plan P3的有效骨筛选，并与前一批镜像指趾/翼语义修复一起验证。现已保留主目录已有迁移内容合并，发布0.3.9、同步运行服务并刷新实际Chrome主窗口。最终验收见[0.3.9交接](monster_video2rig_039_handoff.md)；整个Plan仍继续实施。

## 纠正审计判断

直接蒙皮质量为零，并不足以判定一个Root无效。上游`video2rig/creature.py`将有权重骨骼的祖先纳入有效集合；多个有权重子树共同使用的Root仍提供整体运动。此前“Root下Hips/Head各50%时保留Root就是缺陷”的判断过宽，现撤回该判断。对应探针可以作为保留共同Root的回归例，而非失败证据。

实际缺口是多层无权重组织链（例如`Root→Org→Hips`）仍产生冗余自由度，并且被排除/固定的骨骼在FK中丢失保存的非零`pose_basis`。

## 行为

- 对蒙皮质量占比严格大于0.4%的骨骼及祖先建立有效集合，保留现有Jaw实测支持例外，排除刚性/显式固定骨。占比使用实际权重总量，避免整体权重缩放改变筛选结果。
- 仅当一个活动根组件的所有正权重影响都位于Hips子树内时，移除Hips之上的组织链自由度，并将六个世界运动通道交给Hips。共同根下有其他带权重分支、祖先自身哪怕只有微小正权重，均不提升；显式固定根也不被替换为自由子根。
- 冻结根和冻结子骨保持原始局部旋转/位移。子骨固定偏移参与同一解析链式导数；Hips世界参数从完整保存起始姿态推导。
- V3采样回传全部骨骼的精确局部位移，并直接应用已求得的局部姿态，避免旧写回循环再次清零冻结祖先位置或错误换算提升后的Hips位置。V2采样路径保持原入口。
- 快门关键帧只扰动实际拟合骨。被冻结的组织链/刚性骨保存原始局部姿态，中心帧不再二次限角；挥击和突进分别对自由肢体/运动根施加2.5倍角位移。固定骨在带限角配置的V3烘焙中也按保存局部基保持，而不是强制回到绑定姿态。

## 验证

新增7项测试覆盖多层空链、共同Root、微小祖先权重、显式固定根、权重缩放、非零起始姿态、点/方向雅可比数值差分，以及真实Blender采样/GLB往返。旧实现的空链、缩放及起始姿态反例实际失败。

原生场景包含5根有roll的骨、3个蒙皮网格、非零物体旋转/位移及组织根/刚性子骨的非零局部起始姿态。经过生产采样、带显式限角配置的烘焙、GLB导入，在0与2.5帧比较所有骨矩阵和双向表面距离，最大误差`9.618198032537427e-7`；有效通道9个（Hips世界6 + UpperArm.L局部3）。真实快门关键帧验证冻结骨在0/1/2帧均保持保存局部基，中心全部世界矩阵/表面保持采样值，并用相同随机输入实测STRIKE肢体和LUNGE根角位移为基准的2.5倍。带限角烘焙新增反例在修正前实际出现Rigid1旋转矩阵最大差0.3744，修复后通过。

包含快门/带限角烘焙修复的最终版本：固定0.3.8基线加修复206项、保留迁移内容的固定合并副本216项，以及合入实际主目录后216项全部通过、零跳过；Node73项与历史2K fast的9个golden probe通过。最终构建标识`v0.1.14-20261027061432`已在实际主窗口确认，HTTP/MCP一致。逐文件哈希与发布日志见`var/monster/verification/video2rig_v3_039/`。三处Worker修改的三方合并无冲突，最终源码与验证副本也逐项一致。

## 文件与复现

工作树：`/Users/bugyaluwang/.codex/worktrees/monster-semantic-fixes/codex-game-studio`。

- 实现：`packages/monster/worker/reprojection_solver.py`、`studio_bpy_worker.py`。
- 测试：`packages/monster/tests/test_v3_motion_roots.py`、`fixture_v3_motion_roots.py`。
- 证据：`var/monster/verification/motion-root-fixes/`内的补丁、三方合并内容、逐文件哈希、原生场景与`native/motion-roots.json`。
- `semantic-and-root-fixes.patch`基于不可变0.3.8发布包；`semantic-and-root-fixes-current-main.patch`基于捕获时的主目录，仅执行过`git apply --check`，未应用。

```sh
MONSTER_TEST_BLENDER=/Applications/Blender.app/Contents/MacOS/Blender packages/monster/.venv/bin/python -m unittest discover -s packages/monster/tests -p test_v3_motion_roots.py -v
MONSTER_TEST_BLENDER=/Applications/Blender.app/Contents/MacOS/Blender npm run monster:check
```

此批已完成合并、打包、运行目录同步及实际主窗口验证。原Plan其余未验收项继续保留。
