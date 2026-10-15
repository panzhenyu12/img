# 同侧背景 donor 隔离实验（2026-10-15）

结论：两轮 model-free 实验均拒绝，`accepted=false`。不改 Rust、WASM、Go、前端、服务或生产默认。**model-free scene donor 也不足以恢复本次目标域的真实被遮挡背景，不能接生产。** 此处结论限于本次冻结层/稀疏写入域和这两种取样策略，不表示所有非神经补全方法均不可能。

## 冻结和可写合同

原图为 `test/results/user-face-contour-20261014/1024/canonical-source.png`（1024×1024）。source、graph、V36 inverse 和 v2 source decomposition 固定；alpha、耳朵、金属前景不变。只更换 v2 `original-backing` 中独立支持、实际因形变露出的旧 skin strip 的 backing 颜色。目标域须同时满足：原 write mask、source pure/trusted support、new transported skin alpha < .999、原 unknown/insufficient 均不允许、耳朵/头发保护外、corrected accessory alpha ≤ .05。全部输出保留原图 alpha。

旧 semantic `accessories >= .05` 不是最终归属真值，因其正包含误标皮肤。如果仍整区冻结，旧 source strip 可写像素两个方向都为 **0**，控制组见 `scene-donor-v1/*/old-semantic-control-target.png`。本实验沿用独立 source evidence 推翻误标，未直接把低 accessory alpha 当成皮肤证据。

## 方案与实际结果

| 方案 | +100 | -100 | 结论 |
|---|---:|---:|---|
| v1 同侧 coherent patch，sigma 0/1/2 匹配 | target 216，变化 213 | target 39，变化 29 | 4 个 component/direction 的背景颜色门禁全拒绝 |
| v2 最近同侧背景边界，9 列中值、纵向 sigma .5/1/2 | target 216，变化 213 | target 39，变化 30 | +100 仍大片肤色三角、断续双边；-100 黑楔/锯齿仍在 |

v1 从完全 scene-safe 且 eroded 2px 的同侧背景取完整平移 patch，所有 donor 排除 subject、face/faceOutline/faceContourMask、skin、neck、ears、accessories、hair。多尺度匹配的大窗口被远处白色/棕色场景梯度支配，最优平移仍为左侧 -82px、右侧 +44～46px；RGB RMSE 约 17.7～25.2/255，超过预置门槛 8。`candidate.png` 是诊断结果，`donor-gated-candidate.png` 与 frozen v2 完全相同。

v2 缩小为最近同侧 scene 边界，每行取 9 列中值，并用已观测最内侧 3 列留出验证选择纵向平滑尺度。共使用 1,836 个 scene donor 像素，全部在 scene-safe 内。左侧留出颜色 RMSE 10.26，不通过；右侧 RMSE 3.75，梯度 RMSE .66，可解释已观测边界，但隐蔽目标距离 donor 中位数 44px，不能据此声称隐藏背景被验证。

目标边界梯度（RGB 欧氏距离均值）实际恶化：

| 方向 | frozen v2 | v1 | v2 local |
|---|---:|---:|---:|
| +100 | 23.55 | 73.61 | 82.06 |
| -100 | 10.51 | 10.92 | 11.61 |

这个指标不是“美观分数”，也不能单独认定故障原因；原分辨率图同时确认旧条残留、断续边未解决，故拒绝。

## 重要诊断

+100 仅 component 1 的 174 个、component 2 的 42 个旧 skin 像素允许更换 backing；大片肉眼可见肤色三角/边缘仍不在独立 source support 内。-100 只有 39 个像素属于实际露出的 backing，多数残边不是背景颜色可改变的层。继续在这片稀疏域换 LaMa、Poisson 或黑色填充，都不能修复其外部残留。下一步应验证缺失的 source skin support，并保证 source 清除与 destination 重建作为一个整体，不应以扩大未知 mask 来绕过证据。

## 证据与复核

- `scene-donor-v1/{100,-100}/` 和 `scene-donor-v2-local/{100,-100}/`：完整 `candidate.png`、原尺寸全脸对比、两侧耳饰原尺寸对比、target/donor/保护 overlay、差分、donor 选择数据。
- `scene-donor-v1/report.json`、`scene-donor-v2-local/report.json`：冻结 SHA、像素统计、边界量化。
- `test_scene_donor.py`：4 个结果，区域外/保护/alpha/unknown/insufficient/inverse 不变量；人工污染保护像素和 donor 来源必须触发拒绝；边界恶化不得晋升 accepted。通过不表示视觉合格。

```sh
services/background-worker/.venv/bin/python experiments/accessory-skin-decomposition-v1-20261015/test_scene_donor.py
```

两脚本拒绝覆盖已有输出，保留全部失败证据。实际测试结果：4 行通过，`accepted=false`。
