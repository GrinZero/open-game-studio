# 审计记录、版本与回退

G1/G2/G3 和最终批次验收的判定来源是用户对具体材料的明确反馈。使用现有仍有效的同版本决定，不要求重复确认。用户已清楚选择当前展示候选或批准当前结果时可据实登记，无需要求复述固定口令。

## 保存什么

候选动态材料按 [候选视频审阅规范 v1](candidate-video-review.md) 保存独立视频主入口、同步来源和哈希。新 G1 报告使用 `presentation_contract=monster-candidate-video-review-v1`，完整锁定 4×3 视频引用及证据；历史展示升级另建审阅版本，保留原签名和用户决定。

先读取已有 `stage_state.json`、项目资源版本和用户决定；保存时合并自己的记录，保留历史与 Worker 产生的阶段字段。在用户项目的 `.game-studio/monster/<monster_id>/reviews.json` 与 Drive 的 `stage_state.json` 中同步维护审计账本，并在交接中给出路径。

每次审阅至少保存：

- 关卡 `G1|G2|G3|final_batch`、`monster_id`、项目/资源版本及审阅包位置。
- 输入、完整模型/骨架、采样器与配置、数据集、checkpoint、考试视频或动作批次的实际版本与路径记录，以及锁定的多视图拼合分辨率、单面板分辨率与视角排布（`composite_width × composite_height` / `panel_width × panel_height` / `num_views` / `view_order`）；没有的字段不编造。
- `user_decision=pending|approved|changes_requested|rejected`；等待阶段分别使用 `awaiting_rig_review`、`awaiting_pose_review`、`awaiting_exam_review`、`awaiting_final_batch_review`，生产后按批次和片段保存当前验收状态。
- 用户原始反馈的可追溯引用或原话、决定时间及所针对的版本。`pending` 不写通过时间；没有用户证据不产生 `approved`。用户提出返修时保留原记录并新建后续版本。
- **G1 锁定材料**：唯一的 `selected_candidate_id`、完整绑定模型、解剖学骨骼规范命名校验记录（`bone_naming_validation.valid=True`、零匿名骨、`monster-bone-taxonomy-v3` 映射清单）与 `studio_rig_meta.json`。
- **G2 锁定材料**：固定 `seed`、采样器版本、`joint_limits.json`、异常重抽日志，以及真实 Blender 输出的原始标签、相机、姿态记录、总览图、各视角原图/裁剪图与 2D 叠加图的文件哈希清单（`materials.g2_review_artifacts`，总签名 `g2_review_sha256`）。G2 覆盖审计在实际拟合通道上进行，世界根角度单独记录，冻结骨保留保存的非零 `pose_basis`。
- **G3 锁定材料**：实际 checkpoint 路径、HTML、诊断报告、预测轨迹、原考试视频、同视角/3⁄4/轮廓对比视频与实际烘焙 GLB 的文件哈希清单（`materials.g3_review_artifacts`，总签名 `g3_review_sha256`），以及轨迹输出的 `at_limit` 卡限诊断（非铰链骨超过 25% 帧距关节极限不超过 `2°` 时列出骨名和帧占比，交由用户结合视频判断）。
- **最终批次锁定材料**：当批文件清单 `batch_manifest.json`、非空预测轨迹、通过的标定极线同步门禁、真实 Blender 逐帧回读验证报告（在轨迹 FPS 下核对骨头与世界旋转，使用实际 Action 小数起止帧检查 `:loop` 闭合，`valid=True`）及逐片当前版本的用户决定。

记录里不包含 Worker 密码或私密认证头。Drive 上传和下载记录包含真实文件 ID/路径与大小，确认完成后标记 U1/U2 或归档完成。仅上传脚本配置不产生新的 U2 模型版本。

## 决定的有效性与分级失效规则

门禁绑定实际依赖版本与材料 SHA-256 签名，而非仅看 `monster_id`、文件名或最新标签。批准前及下游门禁均重新核对文件内容与哈希；任一材料变更或缺失时自动触发对应级别的失效，并在账本中追加失效原因（保留历史审批记录备查，不删除历史模型或权重文件）。批次中只要有当前视频仍待审或需返修，就不满足“本批全部满意后释放实例”的条件。

| 改动或失败 | 保留与失效范围 | 下一步 |
| --- | --- | --- |
| 只改 `joint_limits.json` 关节角度限制、随机采样配置或 seed（`sampler_script_or_seed_change`） | 模型和骨架未变，G1 保留；G2 及依赖原采样器/原 `joint_limits.json` 的训练/G3/生产资格失效 | 运行 `joint-limits apply` 更新限制后新抽 50 个边界探索随机姿势，重新提交 G2，通过后再训练 |
| 网格、解剖学骨骼命名或 `kind`/`anatomical_role` 分类、骨骼轴、蒙皮权重、装饰绑定发生实质改动（`local_model_or_bone_naming_change`） | 受影响的 G1 材料需补审；G2、数据集、checkpoint/G3 与最终批次失效 | 本地强制规范重命名并核验 → 受影响 G1 → 50 姿势/G2 → 下一次云端消费前完成 U2 |
| 骨架结构或层级缺陷（`skeleton_topology_defect`） | 选定候选和全部依赖关卡（G1/G2/G3/最终批次）失效 | 重生成四候选，精修骨架与恢复装饰并重新提交 G1/G2 |
| 输入姿态、焊接粘连、严重遮挡（`u1_input_defect`） | 原 U1 与全部下游版本失效 | 预检、新 U1、四候选、G1/G2 |
| 考试视频、标定或重投影求解/后处理参数调整（`exam_video_or_ik_fix`） | 未变模型/采样器的 G1/G2、合成数据集与已有 checkpoint 均保留；仅 G3 重建结果需刷新 | 直接在现有权重上调用 `reexam_only` 用已有视频重考 G3（或重烘焙动作），不重抽数据、不重训模型 |
| 姿态泛化不足（`pose_generalization_insufficient`） | 保留未变模型的 G1；采样分布、训练权重和 G3 需更新 | 回 G2，重新采样、训练并重考 G3 |
| 视频生成分辨率档位（`MONSTER_ENABLE_2K`）、视角模式（`MONSTER_ENABLE_3VIEW` / `MONSTER_STUDIO_LAYOUT`）或相机焦距（`MONSTER_CAMERA_LENS`）变更（`resolution_or_view_mode_change`） | 模型和骨架未变，G1 保留；数据集签名改变，G2/训练集/checkpoint/G3 需按新相机标定刷新 | 按新相机标定与视角排布刷新 G2/合成训练图、训练 Reader 并重考 G3 |
| 仅主干冻结／部分微调策略、学习率、移动窗口或训练轮次变更 | 骨架、采样器和全画幅数据标签未变时保留 G1/G2 与合成集；新 checkpoint 必须重做 G3 | 在独立实验路径训练，按固定留出集选择权重，再用同一独立考试视频复考 G3 |
| Reader 标签契约变更或新开启密集表面场/局部手爪探针（`reader_label_contract_change` / `dense_surface_enabled_0_to_1`） | 模型与骨架未变，保留 G1/G2；数据集签名变更，需重新生成标签并训练对应探针权重 | 重新合成训练标签、训练探针权重并重考 G3 |
| 只改生产动作规格、提示词或该动作 seed | G1/G2 与未变 checkpoint 的 G3 保留；该片及对应批次当前验收需重做 | 保持相同锁定分辨率与标定重做相应动作，等待当前片段的用户确认 |
