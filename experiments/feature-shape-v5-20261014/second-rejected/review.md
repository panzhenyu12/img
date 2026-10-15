# 整眼包络 + screened biharmonic V5 第二候选：失败归档

2026-10-14 实现，2026-10-15 归档。只在独立 Rust crate 研究 eyeTilt / eyeSpacing，正式 geometry.rs 未改。完整 public apply_graph_json 原尺寸用户图两个控制 ±100 均保留。

单独的整眼 polygon 与 iris-sized circle 几何代理通过，不构成图像自然度合格。眼睑包络由原眼检测曲线的邻接估计形成，没有独立检测双眼皮或眼袋真值。

## 原尺寸视觉失败

本代理查看原尺寸/2x 图判失败。eyeSpacing+100 鼻梁两侧有平直暗带/方形边界；-100 虽按连续几何预算分别限制到请求的 50.605% / 50.060%，内眼角附近仍产生新折痕。eyeTilt 双眼及双眼皮旋转更一致，但近鼻梁皮肤仍有挤压。没有进一步扩大照片矩阵；后续 +30 是针对该失败的一项因果诊断。

- [眼距+100 原图/V4b/候选](../second-candidate-fixed/user-eyeSpacing100-comparison.png)
- [眼距-100 原图/V4b/候选](../second-candidate-fixed/user-eyeSpacing-100-comparison.png)
- [倾斜+100 原图/V4b/候选](../second-candidate-fixed/user-eyeTilt100-comparison.png)
- [倾斜-100 原图/V4b/候选](../second-candidate-fixed/user-eyeTilt-100-comparison.png)

## 原断言回归

完整默认 geometry 测试为 **31 pass / 1 fail / 3 ignored**。未删改 V4b 断言。失败仍是 browPeak=-100、browTail=-100、eyeSpacing=-100：额外眉运动对原固定眼包络影响 0.327276px，超过原 0.05px 上限；fold=0、minimum Jacobian=0.455986、reverse error=0.000687px、fixed donor=0。本轮已加入移动眼扫掠保护，但没有完整覆盖回归定义的静态 lid envelope，故仍不符合原组合合同。测试止于第一处失败，不能声称全部 86 场景已通过。

新增整眼测试（两控制 × ±30/±100 × 双眼）通过，最终 inverse rigid error 最大 0.000566px。真实原尺寸 ±100 图 polygon area 误差小、max pose residual <0.003px。这些只能说明当前受限 pose 的眼主体刚性，不能替代周边皮肤效果。

求解采用 f64 PCG，1800 迭代预算，多数分量到预算上限，实际残差 RMS 约 1e-4。没有将预算停止报告为收敛。

## 实现与实现错误

第二候选采用完整邻接估计包络的原眼曲线 SE(2) generator、各眼独立 screened biharmonic vector field、原非凸眉/noseRoot/noseBridge 与对侧扫掠包络固定约束。沿请求射线检查连续轨迹并连续饱和，没有首版的滑杆阈值整项骤然关掉。

第一实现误将非凸眉取 hull，冻结眉弓下方自由皮肤，使眼包络假性冲突而变成 identity；该错误已修正并保留错误日志。随后修正小图 grid 的固定 .75px 下限以保持与原尺寸相对尺度。只将修正后 immutable worker 的输出作为第二版有效证据。

候选源码、immutable worker 信息与验证日志在本目录；全部照片保留在 second-candidate-fixed。原 worker 与源码 hash 见 [worker.json](worker.json)，证据 hash 见 [evidence-hashes.json](evidence-hashes.json)。代码不接正式 Web/API，不宣称 8 分。
