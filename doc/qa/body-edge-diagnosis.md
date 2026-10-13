# 身体轮廓细边归因复核

日期：2026-10-13。该复核只读取 V4 固定渲染结果并在独立 probe target 中导出层，不修改 production binary。

## 结论

`fashion-standing / right-upper-arm` 左侧最明显的亮细线在原图已经存在，V4 输出仍位于原位置，不能把它判为新生成鬼影。进一步导出 `raw-target`、细化后的 `target`、`warped-coverage`、`holes`、F/B 层和供体样本后，发现这条线附近的语义问题来自 part mask：在 `(x=79,y=340)`，目标 alpha 为 1、`hole=true`，但源像素是椅背亮线 `[105,134,142,255]`。收缩时该点被当成袖子并移除，供体使用深色椅背 `[22,47,49,255]`，所以亮线不能继续存在。

这不是一个适合全局调 F/B 的问题。若把 F/B 颜色估计改成更锐利，仍会依据错误 target alpha 抹掉椅背；若冻结整块外侧，又会失去真实袖子轮廓。正确的修复方向是让 partMask 结合主体边界、局部颜色/纹理连续性或用户 raster 校正，剔除与袖子几何不相容的背景细节。原图中的固定亮线不应被算法“修掉”。

`fashion-standing / left-thigh` 的新露出带在 `(x=372..375,y=640)` 使用的背景像素约 `[174,164,26,255]`，与源植物像素一致；该案例的 donor 选择正确。F/B 估计层在软边上会变平滑，但不是这条植物边缘的主要失败源。复杂背景若没有邻近合法 donor，仍需明确报错或手工修正。

## 复现资料

- `.run/body-edge-arm-layers.png`、`.run/body-edge-thigh-layers.png`：原图、原始/细化目标、warped coverage、holes、估计层、填充层和输出的局部放大。
- `.run/body-edge-arm-samples.json`、`.run/body-edge-thigh-samples.json`：仅 probe 构建生成的 changed-pixel 样本，包含源像素、目标 alpha、warped coverage、hole、foreground/background 和填充值。
- 该 probe 复制了 `core/image-core` 到 `.run/body-edge-probe`，仅增加诊断导出；没有将诊断代码纳入 production。
