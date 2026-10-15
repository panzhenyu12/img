# ZIM 按组件验收：固定门槛的语义纠正

日期2026-10-15。复用全部50组独立原照与30组开发原照raw推理，不重跑模型，不改变alpha或任何数值阈值。

原提示指向单个头发连通组件，但验收用整张图hair核心作分母，导致正确的小发束必然被拒绝。新版本按该提示组件的高置信核心计算保留率；其它门槛及官方score选择规则完全不变。旧选择单独保留。

- Holdout：18项abstain→可选（makeup10、dark8），其余32项不变。
- Development：30项选择完全不变，仍有2项abstain。
- dark负点combined-point合并两个原始组件alpha后，全hair参考核心保留85.23%→96.36%；combined-box为74.42%→95.75%。右侧大片卷发恢复，但仍未经过物理边缘/颜色分解及warp验收。
- makeup小发束单独保留99.51%，全图union覆盖基本不变，因大组件输出已经包含该小束。

统计不是alpha精度、视觉通过率或8分。逐组件可选、global union、partial/abstain、未提示组件、软skin alpha分别报告；所有union均标为物理matte未验收。旧mixed的低透明肤色幽影也没有被此语义修正掩盖。

[完整规则与评审](../../experiments/zim-component-aware-20261015/review.md) · [全部80组审计](../../experiments/zim-component-aware-20261015/outputs/report.json)
