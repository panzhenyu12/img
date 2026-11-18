# FUNC-068：眼睛高度源材质准入候选

状态：隔离候选；公共验证入口已在缺少 ocular source evidence 时拒绝非零 `eyeHeight`，默认渲染仍不接入。

2026-11-18：独立实图审查确认 `eyelid` 仍通过通用纵向形变压扁/拉长虹膜，普通 ±30 已约产生 8% 的纵向比例变化。公共入口现同样拒绝非零 `eyelid`，零值保留旧工程的恒等行为；拒绝错误效果不是完成该功能。已补公共图回归，眼睑/眼高继续进行源遮挡分层与 LivePortrait 眼开合模型的有限实图对照，详见 [坏例证据](../qa/feature-eyelid-source-material-review-20261117.md)。

## 目的

`eyeHeight` 不能把 MediaPipe 的眼睛多边形、上睑/睫毛或高光区域直接当成同一种可移动材料。当前通用局部 SVF 没有 ocular source layer 时，会把上睑边缘一起拉动，产生眼裂不自然和睫毛断层。

## 候选规则

`core/image-core/src/eye_material_admission.rs` 对候选源 mask 做三层准入：

1. 必须位于检测到的眼睛 aperture 内，且离 aperture 边界至少给定像素余量；
2. 必须属于足够大的 4 邻域连通分量；
3. 非法尺寸、非有限概率、退化多边形或证据不足时返回全拒绝 mask，不猜测隐藏眼睑材料。

因此，贴着上边界的睫毛/眼线和孤立高光不会自动成为 `eyeHeight` 的可移动 source layer。返回值是二值准入，不代表物理 alpha。

公共 `feature.adjust` 图在没有经过检测协议提供的 ocular source layer 时，会返回“`eyeHeight` has no validated algorithm yet”，避免把通用纵向 SVF 的破图结果导出为成片。该拒绝是质量保护，不代表眼高功能已完成。

## 验证

```text
cargo test --manifest-path core/image-core/Cargo.toml --release --lib geometry::eye_material_admission
```

3 个合成反例通过：上睑边界与孤立睫毛拒绝、畸形/不足证据拒绝、内部 ocular 材质在边界余量下保持连通。

## 尚未解决

候选尚未接入 `eyeHeight`。真实照片还需要 canthus、上下睑有序曲线以及巩膜/虹膜源证据；缺失这些证据时应保持拒绝或仅允许轻度闭合，不能把该候选测试当成眼睛高度可用性证明。
