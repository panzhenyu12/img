# 人脸颈部接缝 v5 局部改进

日期：2026-10-13

## 改动

`restore_neck_lighting` 在旧下颌与新露颈部的语义接缝处增加局部一维颜色桥：只使用同一 `neck` layer 中、已成功观察到的两侧 donor，沿 contour normal 做短距离平均，并按旧 contour 距离衰减。`hair`、`ears`、`accessories`、用户保护、场景和冻结像素不会成为 donor，也不会被写入。

这一步针对 v4 chin=-60/-100 的细亮/暗接缝，保留颈部低频照明重定位和原有高频纹理；它不是全局模糊，也不改变轮廓几何。

## 验证

- Rust release `face::tests`：23 passed，1 ignored。
- 新增 `old_neck_join_is_continuous_and_keeps_protection_and_distant_texture`，核对接缝亮度单调变化、冻结像素和远端纹理不变。
- 6 张真实人像 × `chin/jaw` -100/-60/+60/+100：48/48 hard protection checks passed；报告：`test/results/face-seam-v5-bridge/doc/replay-report.json`。
- `portrait-makeup chin=-100` 局部原 v4 与桥接结果最大 RGB 差 10，写入像素 704；输出需重新人工复核。该改动没有自动授予 8/10；整体人脸质量仍需继续评审。
