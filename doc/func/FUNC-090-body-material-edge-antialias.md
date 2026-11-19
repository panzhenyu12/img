# FUNC-090 身形材料边界抗锯齿

## 目的

低分辨率人体解析 mask 被放大到原图后，腰部收窄会把二值衣物边界直接带入连续位移场。白衬衣与深色外套交界处因此出现阶梯边缘，即使位移场本身没有折叠。

## 实现

`core/image-core/src/body.rs` 的无 `opticalAlpha` 回退路径现在对 `partMask ∩ subjectMask` 生成一个受 subject 约束的 3×3 双边 coverage 过渡：

- 只处理语义边界，不改变材料内部的 alpha；
- 使用源图颜色相似度抑制跨衣物、手和背景的颜色泄漏；
- 输出 alpha 始终不超过 subject ownership；
- 显式 `opticalAlpha` 路径不做修改，保留调用方提供的材料真值。

这一步只改善采样边缘的覆盖连续性，不负责修复错误的服饰实例归属；归属仍由前端解析器和 source-layer 接口负责。

## 验证

- `coarse_material_boundary_gets_a_bounded_antialias_transition`：边界产生有限 coverage，内部仍为 1；
- `material_boundary_never_exceeds_subject_ownership`：subject 为 0 时输出全部为 0；
- `fashion-standing` 原图 `waist=-30` native replay 已重新生成临时输出 `/tmp/fashion-soft-oldmask-30.png`，原图、alpha 通道和保护区合同保持不变。

## 限制

解析器完全漏掉的衣物实例仍不能由本步骤恢复；同色遮挡、透明布料和大范围缺失背景需要独立 `opticalAlpha` 或局部补全层。
