# 源图解剖间距约束版：可继续评审，未接生产

回到第三版的联合 SVF，保留整眼刚性/正反眼睑保护，并加膜应变项。直接端点映射已因选区边界折叠拒绝。

## 可行性规则

逐眼检查原始 eye aperture 沿整个请求 SE(2) 路径与 leftBrow/rightBrow/noseRoot/noseBridge 的间距。最小间距暂按原间距 50% 与眼宽 3% 的较大值（不超过原间距）；这是明确可审的工程解剖启发式，**并非学习到的美学先验或已验证最优比例**。沿请求射线找首次违反约束处，在其 80% 后连续趋近，避免突然切断或按 abs(amount)<=30 切换算法。

输出 requested、applied、poseScale、feasiblePoseScale、poseLimitReason，以及逐部位的原间距/阈值/请求间距/实际所用轨迹间距。`feasiblePoseScale=4` 代表射线搜索到 4 倍没有遇到该约束，并非证明全局可行最大值恰为 4。

这张图 eyeSpacing -30 的两眼完整执行；-100 由 noseRoot 限制到左 53.6948%、右 53.9491%，平均实际平移约 10.195 px。eyeSpacing +100、eyeTilt 各方向在这张图未受该间距约束限制。

## 证据

- 同一用户 1024×1024 原图、原检测，0 与 eyeSpacing/eyeTilt 各 ±30/±100；零值完整 RGBA 与源图相同。
- geometry **33 passed / 0 failed / 3 ignored**，205.04s。包括原 86 场景、原跨零回归、完整眼形/虹膜几何代理，以及新眼选区/保护边界回归。所有旧断言保留。
- 匹配平均实际位移的 V4b 对照：旧滑杆 -55.5576877，平均位移误差 -0.000656 px，单眼 +0.02378 / -0.02509 px。仪器化旧路径与冻结 V4b native 全 RGBA 相同。见 `matched-control-result.json` 与 `matched-control-comparison-2x.png`。
- 同位移 V4b 的 eye polygon 边段最大相对长度变化为 24.91% / 23.76%，候选 0；这只量化眼形保真，不替代图像美观。

## 不接受部署的剩余问题

候选依然会牵动眉周和鼻梁；强档眼距向外的鼻侧明暗过渡仍显得平。鼻根/鼻梁是软锚，当前可行性数值比较的是移动眼与原鼻区域，而非已经量测最终变形后的鼻区域宽度；因此不能把源间距阈值描述成最终皮肤间距的严格保证。单张图不足以支持评分 8。需独立视觉审查、实际鼻形/眉形保真、其它源图与用户保护组合后再决定方向。

`feasible-source` + `worker-feasible.json` 对应运行完整 33 项回归的源码/worker。当前 `core` 另外只有 test-only 的 `EYE_V6_BASELINE` 仪器化旧路径开关，用于同位移旧对照；默认路径仍为候选，不改生产。`feasible-summary.json` 保存每档的限制、实际输出、求解残差与时长。
