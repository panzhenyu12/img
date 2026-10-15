# 源皮肤 F/B 一致性验证（2026-10-15）

结论：**未验收，不接生产，不达到 8 分。** 修复了前景反解与最终合成使用不同背景的实现 bug；固定几何下的真实照片仍有细线、有限背景域接缝和不自然的脸颈轮廓。后续应建立统一的源图层/未知背景/耳根接口表示，不能把这次局部数值修正算作完整的瘦脸质量提升。

完整代码、输入合同、可重放产物见 [实验说明](../../experiments/source-skin-color-consistency-v1-20261015/README.md)。本轮只修改实验目录，没有更新应用、WASM 或运行服务。

## 控制变量

沿用 V37 原始 physical alpha、inverse/geometry、0/±100 强度和保护域。源图为用户 1024×1024 样例，含 AI 创作标记。B 先验只用已审计 `outputs-v2-consistent/scene-support.npz`，finite 31,665 像素。未扩大 scene/skin 域，未更改耳饰阈值，没有新模型调用。

已观测源 scene RGB 只从同侧真实可见背景提取，独立 skin/accessory 与耳朵/头发/五官排除后要求 RGB 连通。未知处通过局部有效均值和最近同侧 donor 延拓，这是背景颜色先验，不是真实隐藏场景。

## 被测试的算法

只在有限、可信皮肤软边反解 `F*=(C-(1-A)B)/A`。alpha<.1 不反解；颜色受半径 6px 内可信皮肤 RGB min/max±1 byte 限制。颜色投影若令任一源 RGB 残差增大，拒绝整个像素。仅修正 F 的颜色，保持 alpha、几何与原始采样位置。

最终仍使用 `Cout=C+sample(F*A)-F*A+(A-targetA)*B`，且反解、拒绝和合成必须使用同一个最终 B。

## 三版与真实回归

- 初版局部颜色盒能把原本拟合更好的 F 推远，导致局部变暗，拒绝并保留。
- v2 增加单调拒绝；445 个候选纠正 218，拒绝 207。独立审计发现用原 scene B 反解、用经过边界过渡的 B 合成，13 个像素实际源残差变差，最大 4.09 byte。
- v3 只修复这个 B 参数不一致，没有新增阈值。先将真实 13 像素纳入回归并观察红灯，再使用最终 B 取得绿灯。0/±100 都重新重放并检查实际成图。

| v3 指标 | +100 | -100 | 0 |
|---|---:|---:|---:|
| 源候选域 | 445 | 445 | 445 |
| 源 F 字节变化 | 233 | 233 | 226 |
| 实际 final B 下量化后残差增大 | 0 | 0 | 0 |
| 输出相对 B-only 变化 | 445 | 428 | 0 |
| 输出相对 v2 变化 | 46 | 49 | 0 |
| 输出域外 / frozen / opaque / alpha 变化 | 0 | 0 | 0 |

0 图片逐字节等于原图。已确认原始 F 的 inverse 重放 sample、target alpha 与 Rust trace 完全一致；source alpha、geometry、inverse、final B、输出 SHA 都有独立核对。已有 8 项单元/集成测试全通过，额外保存图复核也通过。

已知纹理合成例：错误 F/B 误差 181.05 byte；只纠正 B 后 11.48；F+B 一致后约 2.8e-14。该例假定 A/B 已知，只证明条件性正确；不能据此推断真实 alpha/隐藏 B 正确。

## 原尺寸视觉核验

已看 0/±100 原尺寸全脸，以及 ±100 左/右轮廓，未用缩图掩盖问题。宽灰条的明显改善来自前一轮 B-only；v3 与 v2 的主要区别只是 46/49 个像素的颜色一致性修正。

| 对照 | 判断 |
|---|---|
| 旧轮廓细线 | 仍可见，未解决 |
| finite B context 下沿 | 仍有衔接异常，8px 域内过渡不足以消除 |
| 脸颊—下颌—颈部曲线 | 仍不自然，当前固定几何实验不解决此项 |
| 耳饰、不动区、0 强度 | 本轮字节不变量通过，不代表整张成图可用 |

原尺寸证据：

- [+100 全脸，原图 / B-only / v3](../../experiments/source-skin-color-consistency-v1-20261015/qa/final-verification/100-face-native-1x.png)
- [-100 全脸，原图 / B-only / v3](../../experiments/source-skin-color-consistency-v1-20261015/qa/final-verification/-100-face-native-1x.png)
- [0 全脸](../../experiments/source-skin-color-consistency-v1-20261015/qa/final-verification/0-face-native-1x.png)
- [v3 原始报告](../../experiments/source-skin-color-consistency-v1-20261015/outputs-v3-final-b/report.json)
- [独立保存图验证](../../experiments/source-skin-color-consistency-v1-20261015/qa/final-verification/report.json)
- [最终 B 不一致红灯](../../experiments/source-skin-color-consistency-v1-20261015/qa/final-b-red.log) / [8 项绿灯](../../experiments/source-skin-color-consistency-v1-20261015/qa/final-b-green.log)

## 可保留模块与边界

保留 `decompose.local_prior` 的可追溯源背景估计、`solver.solve` 的局部颜色约束与单调拒绝、`pipeline.correct_foreground` 的唯一 final B 参数、`compositor` 的 native f32 重放，以及真实集成反例。它们为后续统一表示提供可验证组件，不能单独作为生产瘦脸方案。

源拟合单调不等于视觉自然，也不等于 alpha/B 物理真实。残差合成会保留源模型误差，错误背景域依然能留下旧轮廓；小 alpha 拒绝区和材质不确定区没有足够证据，不能靠扩大阈值抹去。最终 byte 量化的单调性已在本次保存样本验证，任意新图仍须另测。

实验到唯一 v3 修复为止收束，所有失败候选保留，不继续调整旧 trace 的阈值。
