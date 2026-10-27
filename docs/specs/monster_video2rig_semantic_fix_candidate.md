# 匿名骨架语义修复（已合入0.3.9）

本修复推进原Plan P2的镜像指趾和翼语义，不宣称整个Plan完成。最初在独立受管工作树验证，现已保留当时主目录全部迁移内容合并，发布为0.3.9并同步运行服务、刷新实际主窗口。最终216项检查与发布证据见[0.3.9交接](monster_video2rig_039_handoff.md)。下述199/209项等数字为隔离阶段的历史验证。

## 修改与行为

- 指趾按角色左右镜像的内外方向编号，并以指链远端几何区分共起点指扇；保留现有group编码和已知v3名称。相同左右指/趾对应相同基名，而不是内指与外指互配。
- 几何测得的翼kind/flatness/winged/role先于掌部多分支分类生效；翼掌仍可识别下辖指骨，并沿单链传播至近端，遇躯干或其他语义分支停止。旧v2命名的原始kind/group/index仍可只读解析；几何新标记的旧LIMB可升级为Wing。已有明确v3骨名优先于过期提示。
- 候选最终重命名前保留prepared_bones的几何语义，修复中间重新抽取元数据时丢失翼类型的问题。仅改分类器不足以贯穿真实候选入口，因此一并修正交接。
- 默认限角/耦合随新语义重算，canonical_default不再被当作用户自定义；用户显式来源的限角和耦合保持。

## 验证范围

- 新反例在旧基线实际失败：左右Finger21/23反配、共头指扇次序错误、8根翼骨变成普通手臂。修复后7项新增测试通过。
- 原生Blender使用43根实际骨、两个网格、顶点组及两个Action关键帧；生产候选重命名保留8根Wing、所有左右指趾配对和父关系，表面/骨矩阵最大差为0，权重仍为原始1.0。翼默认X/Z限角也按新语义更新。
- 0.3.8固定发布基线+修复：完整check **199项通过、零跳过**，包含原生Blender、Notebook/API、训练重载和P1–P4；Node **73项通过**；历史2K fast **9个golden probe PASS**；build/distribution通过。构建仅在隔离树，标识v0.1.14-20261027052126，没有替代主窗口版本。
- 同目录另一执行者后来修改了相同命名文件（新增v3 feature flag和Jaw探针），基于共同0.3.8祖先的三方合并无冲突，保留这些逻辑。捕获当时主目录全部包文件，合并修复后执行同一完整check：**209项通过、零跳过**。这证明该固定副本兼容，不表示后续继续变化的主目录也已验证。

## 文件与复现

工作树：`/Users/bugyaluwang/.codex/worktrees/monster-semantic-fixes/codex-game-studio`。

- 对发布基线的补丁：`var/monster/verification/anatomy-fixes/semantic-fixes.patch`。
- 对捕获的主目录版本的补丁：同目录`semantic-fixes-current-main.patch`；当时`git apply --check`通过，仅检查未应用。合并时须重新检查当前源码，保留后续变动。
- 基线/候选/主目录快照逐文件哈希、原生数据和检查日志在同目录`candidate-manifest.json`、`latest-compatibility-manifest.json`、`verification-summary.json`。原主目录也保留副本`var/monster/verification/monster_anatomy_isolated/`。
- 新测试：`packages/monster/tests/test_v3_anatomy.py`与`fixture_v3_anatomy.py`。

原生测试命令：

```sh
MONSTER_TEST_BLENDER=/Applications/Blender.app/Contents/MacOS/Blender packages/monster/.venv/bin/python -m unittest discover -s packages/monster/tests -p test_v3_anatomy.py -v
MONSTER_TEST_BLENDER=/Applications/Blender.app/Contents/MacOS/Blender npm run monster:check
```

## 未完成与继续

组织根的后续判断与修复见[空组织链与起始姿态](monster_video2rig_motion_root_candidate.md)：零直接质量的共同Root应保留，实际修复针对Hips之上的多层无权重组织链及冻结起始姿态。本批已热更新为0.3.9；原Plan的其他缺项继续保留。合并前已确认运行目录与捕获的主目录基线逐项一致，保留所有迁移内容；没有根据AGY界面超时推断任务结束，也未重启/终止未知进程。
