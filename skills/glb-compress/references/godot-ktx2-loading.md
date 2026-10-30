# Godot KTX2 加载适配

适用于运行时通过 `GLTFDocument` 加载本工具输出的 KTX2 GLB。KTX2 继续默认开启；先保证显示链路正确，再按 [人工对比流程](ktx2-human-review.md) 等用户选择。此适配不是用户对压缩画质的认可。

## 使用随附加载器

将 [godot_ktx2_loader.gd](../scripts/godot_ktx2_loader.gd) 复制到目标 Godot 项目的脚本位置。在原来调用 `append_from_file` / `generate_scene` 的入口使用：

```gdscript
const Ktx2Loader = preload("res://your_path/godot_ktx2_loader.gd")

func load_candidate(path: String) -> void:
    var result := Ktx2Loader.new().load_glb(path)
    if result.error != OK:
        push_error("GLB error %s: %s" % [result.error, result.loader])
        return
    add_child(result.scene)
    print(result.loader) # mode, engine, renderer
    # 这里仍为 awaiting_user_comparison；没有自动批准或替换生产资产。
```

加载器内置一张独立制作的 4×4 sRGB 灰色 KTX2 探针。它先检测当前 Godot 的实际解码行为：

- 解码值已保持 sRGB：直通，不应用补偿。
- 解码值被提前转为 linear：注册临时 `GLTFDocumentExtension`，仅对 DFD 标为 sRGB 的 KTX2 颜色图补偿一次；linear 法线/ORM 数据图不变。
- 加载器不可用或结果不匹配这两种已知行为：返回具体错误，不猜测 gamma。

扩展在图片解析阶段补偿，随后由正常 glTF 材质绑定使用，不复制一整套材质/颜色纹理来掩盖色差。注册仅包围一次同步 `append_from_file`，随后立即注销；该区间不要 await 或并发修改全局 glTF 扩展注册表。扩展配置使用导出属性，以适配 Godot 对扩展实例的复制。

GLB、原 KTX2 图像和 sRGB 标记保持不变。不要为了绕过这个加载问题而把交付文件的颜色图伪标为 linear，或烘焙整体提亮。

## 原因与适用边界

Godot 4.7.2 本地复现与 [4.7-stable KTX 加载源码](https://github.com/godotengine/godot/blob/4.7-stable/modules/ktx/texture_loader_ktx.cpp)一致：sRGB Basis 图转为 RGBA32，加载器调用 `srgb_to_linear()`；[glTF KTX 扩展](https://github.com/godotengine/godot/blob/4.7-stable/modules/gltf/extensions/gltf_document_extension_texture_ktx.cpp)直接使用此加载器。StandardMaterial 的颜色采样链进一步解释这批已线性化像素，出现明显变暗、偏深黄。

公开的 [GLTFDocumentExtension 接口](https://docs.godotengine.org/en/stable/classes/class_gltfdocumentextension.html)提供图片解析扩展点；随附脚本利用该入口并通过运行时探针避免在已修复的加载器上重复补偿。

这是当前受影响加载路径的兼容适配，不是 KTX2 编码参数调整。对已被量化成 8 位 linear 的数据做逆变换不能恢复全部暗部精度，因此需保留近景/暗部对照。其他引擎不使用此补偿。

本脚本入口只覆盖运行时 `GLTFDocument` 加载；不会自动修改编辑器的 `.glb` 导入插件或资源导入缓存。正式使用编辑器导入、导出包或新 Godot 版本时，必须在实际入口重测，不把运行时成功直接当作整个项目兼容。

## 效果、资源与人工选择

验证需要直接读取实际候选，比较原版与适配后的结果；修复色彩链问题后，才评估压缩造成的纹理细节变化。记录 `result.loader`、Godot 版本、渲染器、截图与候选文件信息，交给用户选择采用、保留原版或调整重试。候选改变后重新确认。

既有栏墙试验：原 GLB 13.05 MiB → KTX2 2.19 MiB，均保留 4K 和 978 三角形。加载后补偿的早期原型，Forward+ 同镜头前景 RGB 差由约 49.19/255 降至 1.55/255。这是单资产证据，不是跨模型验收线。

文件体积、纹理显存和帧率分开报告。当前 Godot sRGB 路径会解码为 RGBA32；本脚本不承诺颜色纹理仍驻留为 GPU 压缩格式。早期原型复制材质、同时保留旧颜色纹理，显存计数从 192 MiB 降到 160 MiB；随附脚本改为解析阶段处理。当前栏墙实测在 Forward+ 和 Mobile 桌面端均为 192 MiB → 96 MiB（分别扣除空场景基线），避免了早期原型的额外旧纹理；这仍是单资产、无 mipmap 的结果，不能直接当作所有项目或真机的收益。不要沿用另一个实现或未校验的计数来承诺收益。

当前脚本的 [独立统计记录](godot-ktx2-validation.json)包含两种渲染器各八个视图的误差与纹理显存计数；当前编码器生成的三张 KTX2 均通过 KTX glTF BasisU 规范校验。

完整本地证据目录：`var/glb-compress/ktx2-hitl-default-20261030/`（诊断/早期原型）及 `var/glb-compress/godot-ktx-loader-20261030/`（随附脚本验证）。外部分发包含脚本和探针，不包含用户模型或图片。
