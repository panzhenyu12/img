# 五官 eyeHeight 真实入口审计（2026-11-20）

## 结论

当前 `feature.adjust` 的 `eyeHeight` 仍不能算可用。公共图层从浏览器检测结果只携带 `semanticMasks.leftEye/rightEye`，没有 `eyeGeometry` 中的虹膜中心与四点边界；Rust 校验在渲染前明确拒绝非零请求：

```text
feature.adjust.eyeHeight has no validated algorithm without measured eyeGeometry
```

这不是“有变化但效果不好”，而是滑杆请求直接失败，属于 P0 功能缺口。

## 证据

`core/image-core/tests/algorithm_contracts.rs::ocular_occlusion_controls_do_not_export_the_generic_iris_stretch` 对 `eyeHeight=-60/-30/+30/+60` 均验证该拒绝；`core/image-core/src/geometry.rs` 虽然已有 `eye_height_material` 的材料保护分支，但它只在参数包含 `eyeGeometry.<eye>.iris` 时启用。

## 根因

仅靠眼睛外轮廓无法知道虹膜、眼白、上/下眼睑的真实材料边界。直接解除校验会把普通眼部形变应用到整块眼睛多边形，造成虹膜拉伸、眼睑断裂和睫毛变形。当前应先在浏览器/服务端检测阶段输出可校验的虹膜四点几何，再接入已有 `eye_width_material(..., axis_vertical=true)` 路径；在此之前保留显式拒绝，避免生成不可用图片。

## 下一步

1. 在 face parsing worker 中增加 iris center + boundary 四点输出，并在 graph 中写入 `eyeGeometry`。
2. 对正负 `eyeHeight` 真实照片逐眼检查虹膜、眼白、上下睑和睫毛的材料连续性。
3. 通过后再移除该拒绝条件；不得用泛化椭圆参数绕过材料校验。
