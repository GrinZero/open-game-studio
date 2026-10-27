# Monster 0.3.15：训练集 p95 窗口与推理尺度契约

原始video2rig/reader/dense.py的window_side统计整批训练图片的关键点范围，再取p95乘1.12；原实现却在训练时用各帧完整表皮包围盒，在推理时用单帧点坐标分位数，未共享训练窗口尺度。

现在ReaderDataset对全体训练图片中画幅内的关键点范围统计p95（每图至少三点），乘1.12，四舍五入并限制到面板短边。窗口中心跟随各图几何，所有图片共享边长；原全表皮框保留为诊断，不丢失原标签。checkpoint保存reader_window与monster-reader-training-p95-window-v1，包括边长及训练面板尺寸。推理按面板高度比例换算边长，只让中心跟随粗读轨迹，失读位置先时间插值再平滑，不让窗口大小逐帧抖动。RGB、热图上下文、密集场与像素回映使用同一变换。

旧v2路径保留。缺少窗口契约的早期v3权重明确要求重训，原文件保留；具有当前起始姿态/上下文/采样契约的全画幅标签可复用，窗口统计不要求重复Blender渲染。没有重训或替换真实用户角色权重。

验收包含21张图（20张100–120px范围、1张300px极端图）的p95边长精确134px，896→448换算为67px，短边截断及缺失/无效窗口拒绝。既有五帧序列加强为半周转身、横/竖非方形面板、边缘移动与遮挡组合；固定窗口180px、八次TTA和两帧上下文正确，可见点误差≤4px。实际原生横排/堆叠ReaderDataset与训练checkpoint逐项核对同一窗口，生产推理入口确实传递保存的统计。

另持久保存本地TinyBackbone的两轮训练checkpoint/reader.pt，复用0.3.14原生堆叠全画幅标签，实测边长591px/896面板，window.json与checkpoint回读一致。这验证训练/重载接口，不宣称用户角色或预训练DINOv2的GPU质量。

主目录245项Python检查、零跳过（133.410秒），Node73项、历史九探针、打包/build/distribution通过。0.3.15已同步实际runtime，HTTP/MCP和原Chrome主窗口均验证`v0.1.14-20261027084805`，27工具正常，76个发布源文件逐项匹配；两个项目登记SHA256保持`3517552ab1f6836d1fe979c2b263f156f0faa5236624f2a415323b731ba55392`。

证据在`var/monster/verification/video2rig_v3_0315/verification.json`、window.json、reader.pt与日志。不可变包`published/monster-0.3.15.tar.gz` SHA256：`f448dcd5daedcb533177352594a5195902399fb7892ff90788d32a7c12dff82f`。原Plan剩余G2实际标签/相机衔接与边界采样起始姿态、SOP契约验收继续实施；另已证实旧G2边界路径会改动冻结非零basis，反例保留在受管工作树var/monster/verification/sop-v3。

后续[0.3.16](monster_video2rig_0316_handoff.md)已完成SOP实际标签/相机、边界p0、v3数学检查与GLB回读，并热更新。本文保留0.3.15历史快照。
