# 静态眼睑包络轨迹保护：可隔离修复范围

`lid-trajectory-only.patch` 只包含 V6 第二版到第三版的保护域修复。刚性眼核心、眼实际请求、求解矩阵和眼位移输出完全不变。保护域改为原静态 lid envelope 的正反 SE(2) 轨迹并集，修复眼睛在合成映射中离开初始保护区后被眉毛速度二次扭曲的问题。

第三版完整 geometry 回归为 **32 passed / 0 failed / 3 ignored**，日志 `geometry-tests-third.log`。源码保存在 `third-source`；不可变 worker 见 `worker-third.json`。没有删改旧 0.05 px 眼包络偏移、跨零、Jacobian、供体等断言。

用户同源原尺寸 eyeSpacing/eyeTilt 各 ±100：修改前后四个 PNG SHA256 全相同、完整 RGBA 差异均为 0。见 `lid-trajectory-only-parity.json`。

**不能直接当作生产 V4b 的补丁**：生产源码没有 `ElasticSupport` 或刚性整眼扫掠，现有生产 86 场景已经通过。该问题是在 V5/V6 整眼刚性平移扩大实际眼运动之后引入的。将其移植到 V4b 需要先找到 V4b 实际组合失败证据，并针对 V4b 的局部缩放/旋转实际流重新构造保护轨迹。不能为了交付将整个 V6 引入生产。

本补丁修复功能间串扰，不修复 V6 的鼻根细线或眉鼻审美问题。因此第三版整体仍不接受部署。
