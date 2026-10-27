# 0.3.8发布与并发修改交接

后续状态：本文件记录0.3.8阶段历史；语义/空组织链/起始姿态等修复已在保留迁移内容后合入并发布0.3.9，最终216项检查及实际主窗口热更新见[0.3.9交接](monster_video2rig_039_handoff.md)。原Plan未完成。

原始目标仍为`monster_video2rig_integration_plan.md`的七阶段完整落地；未完成，未调用goal complete。

## 当前可用发布

- Monster 0.3.8；工作台`v0.1.14-20261027030844`。
- 表皮点骨名映射/真实叶骨判断、完整附加槽位与扭转观测验证、12种实际非方形相机换算已发布。
- 固定发布内容的完整check：192项通过、零跳过；Node73、历史基准9项通过，build/distribution通过。真实Chrome主窗口已刷新。
- 发布包SHA256：`a6597e6bd84135c78c1a0079995b9e92155d23f39f5481811f1761e1e4192047`。发布76文件与检查快照、runtime一致；27Worker逐一校验。
- 证据根：`var/monster/verification/video2rig_v3_038/`。`release-verification.json`、`published-files-sha256.json`、`runtime-verification.json`与`worker-import.json`为权威逐文件/运行证据；截图`main-workbench.png`。
- 固定检查目录：`/tmp/monster-v3-038-release-validation/packages/monster`，其代码取自已发布runtime并与tar包逐文件校验，依赖环境通过符号链接指向原有venv；没有复制/操作角色或云端数据。

## 尚未发布的修改

只读确认Antigravity另一会话 **Human Differences Beyond Skeletal Structure**，ID `5434b36e-423d-4882-8eb0-f68562755905`，仍在Working，界面连续显示Reader/训练器/Blender等文件Edited。最初计划会话`73e5d078-e637-4e1e-b41d-c65bd2baf620`当前不显示执行中。未停止任何未知进程，也未向另一会话发消息。

写入仍在继续，清单是有时间的快照：`pending-source-differences.json`记录发布/源码哈希与修改时间，`pending-source-snapshot/worker`和`tests`保存当时原文件及unified diff。当前至少包括dinov2_pose_trainer.py、reader_v3.py、reading_pipeline.py、reprojection_solver.py、studio_bpy_worker.py、video2rig_blender.py、video2rig_pipeline.py及test_v3_solver.py。不要直接再次运行整体sync-runtime将这些未验收内容发布。

- 第一次并发源码check：192项，地面回归失败，日志`monster-v3-038-check-merged.log`。
- 本聊天已修正求解器model遮挡先验覆盖当前高置信读数的问题，补`test_prior_capsule_occlusion_keeps_confident_current_readings`；solver9项针对性通过，日志`monster-v3-038-solver-merged.log`。这项后续源码修正/测试尚未发布。
- 再次源码check：193项，远侧置信度边界失败（front=.9、side=.5、residual=.2时被降为.05，应保留.5），日志`monster-v3-038-check-final.log`；验证期间reading_pipeline.py和video2rig_pipeline.py又发生变化。没有修改测试期望掩盖该差异。
- 新增Jaw/持握武器/model可见性/端点动态配对/双向消歧/新训练入口等均不能继承旧测试证明；部分可能属于另一会话更广迁移目标。

## 继续工作条件与顺序

保持双方所有现有修改，不终止未知任务；并发期间在隔离工作树实施和验证，避免对活动源码整体覆盖或热同步。合并时重新捕获当前内容，保留另一执行者的修改，并验证最终合并版本，再执行打包/build/distribution及实际主窗口热更新。用户已授权原Plan实施；本交接文档不新增必须由用户确认的批准门槛。继续原完成度复核中未完成的形态/采样/训练/TTA/SOP门禁工作，不将0.3.8批次或并发阻碍视为整个目标完成。

## 后续只读复核（2026-10-27）

- Antigravity同一会话仍在执行，已看到其运行check失败后检查对应测试与读数代码；未干预会话。
- 当前固定源码快照的标定7项、后处理17项通过；先前side=.5回归已在该快照消失。此结果不证明整个迁移完成或发布。
- 新增独立命名/参数分配探针：镜像指趾编号反向、掌部检测覆盖显式翼类型。此前把多子树、零直接质量Root获6DoF也判作缺陷的结论已撤回：它仍是有权重子树的有效共同祖先。真实缺口为Hips上方多层空组织链和冻结起始姿态丢失，见后续候选。
- 原始输入、输出、实际双生配对、源码哈希与可重跑脚本在`var/monster/verification/video2rig_readonly_audit/`，审计文档`monster_video2rig_readonly_audit_20261027.md`。没有修改生产源码。
- 为防另一执行者重打同名发布包，原已验证tar另存于`var/monster/verification/video2rig_v3_038/published/monster-0.3.8.tar.gz`，SHA256与原清单一致。

## 独立修复候选

已建立受管工作树`/Users/bugyaluwang/.codex/worktrees/monster-semantic-fixes/codex-game-studio`，以76文件逐一匹配的0.3.8发布包为基线；没有新开执行聊天或子代理。完成镜像指趾/翼语义/候选交接和默认限角的隔离修复，新增7项测试及原生43骨命名反例。基线完整199项、当前捕获迁移+修复完整209项均通过；Node73、历史golden9、build/distribution通过。

主目录曾又修改同一命名文件，因此提供保留feature flag/Jaw探针的三方合并补丁，未对活动主目录应用。补丁、双版本哈希、原生结果和日志在主目录`var/monster/verification/monster_anatomy_isolated/`，说明见`monster_video2rig_semantic_fix_candidate.md`。这批代码尚未合并或热更新；AGY界面读取超时不代表其任务结束。后续继续可在隔离树推进其他待办，发布前须整合当时完整源码，避免覆盖迁移中的七个Worker。

随后补充空组织链/冻结起始姿态/采样位移写回修复；两批候选基于发布包完整206项通过、零跳过，Node73和历史9项通过，隔离build/distribution通过。原生5骨/3网格的生产采样、GLB烘焙/导入及0/2.5帧全部骨矩阵/表面最大差9.62e-7。详见`monster_video2rig_motion_root_candidate.md`；此批同样未发布。
