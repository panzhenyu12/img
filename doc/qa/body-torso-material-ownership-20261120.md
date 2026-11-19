# 躯干服饰材料归属回归（2026-11-20）

## 修复

`deriveBodyLayers` 的躯干种子原先允许 `top/dress/skirt/pants/belt/torso` 全部互相穿越。解析器把白衬衣和后挂深色外套都标成 `top` 时，最短路径会绕过衬衣边缘，把外套纳入 waist 的运动材料，真实照片上表现为衬衣下摆台阶和外套被拉动。

本次改动只在前端解析层完成两件事：

1. 从躯干骨轴种子中选出占主导的语义材料类，并要求扩张像素仍匹配该类；
2. 对躯干 owner 增加观测 RGB 边界约束。归一化 RGB 欧氏距离大于 `0.20` 的相邻像素不允许跨越，防止同一 `top` 类中的后挂衣物绕路进入躯干。

该约束不会改变姿态、区域或用户调整强度，也不会把矩形 ROI 当作前景。没有足够的语义种子时保留原来的兼容路径。

## 证据

- 真实 fresh source：`test/results/body-fashion-source-20261119/fashion-standing/source.png`；同一次 FASHN 推理的 `fashn-classes.u8` 和原始 Pose 点作为输入。
- 修改前 torso ownership 覆盖约 `18,473` 个解析像素，并延伸到颈部、腰带与左侧后挂外套；修改后为 `14,504` 个像素，白衬衣主体连续，颈部、腰带和外套不再进入 waist 材料域。该统计来自只读浏览器回放 `/tmp/body-mask-overlay.png`，没有重新推理。
- `apps/web/tests/body-parts.spec.ts`：4/4 通过，包括新增的“同一 `top` 语义下深色后挂衣物不进入躯干”回归。
- `npm run build`：通过。

## 限制

这是材料归属的安全收缩，不能证明所有服饰实例都能自动分层。多件同色相、无清晰光学边界的衣物仍需要独立 `opticalAlpha` 或人工保护层；服务端 `body.rs` 的材料域合成另有独立回归。
