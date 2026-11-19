# 源背景 4px 上下文保护的有界验证（2026-11-19）

结论：在 portrait-614810 胡茬男脸上，固定 4px 上下文保护明显减轻旧轮廓暗带，但原写入域外仍有细线；不能宣称“无缝”或达到可用。portrait-762020 长发女脸的同集合对照未见新增明显退化，但实际改动很小，不能扩大为泛化结论。

## 冻结协议与对照

- 沿用已有 LaMa 权重、256 tile、128 stride、原 mask 内 32px 接受带、sin² 拼接权重；2 CPU 线程。只将模型输入 unknown 按 scipy 默认十字连接扩大 4 次，原输出写入 mask 不变。
- 首次实验遇到扩大后全 unknown 的 tile 而退出。新实验按原始源 mask 预先排除无可读上下文的 tile，并在任何推理前验证原 32px 接受带完整覆盖。旧失败输出保留。
- baseline0 同样从缓存按新的有效 tile 集合与权重重合成，避免把 tile 集合变化混入 guard 效果。614810 排除原 index1，保留 23 个；762020 保留 24 个。
- 两图均从普通 `face-source-replay` rlib 入口回放 0、±30、±60；同一源、geometry、输入 alpha、运动许可。未改 core/worker，未做整体构建。
- 固定输入 alpha 不等于最终消费 alpha 不变：当前入口 `scene-lower-bound` 会随背景 B 改变覆盖下界，因此不能称最终输出为纯 B 单变量传输。

两图的 matched0 与旧 alltiles0 五档输出 RGBA 完全相同；guard4 的零档与源图 RGBA 完全相同。以下差异仅说明改动大小，不是质量评分：

| 样例 | +30 改动像素/最大字节差 | +60 | -30 | -60 |
|---|---:|---:|---:|---:|
| 614810 | 14,418 / 58 | 21,075 / 68 | 8,864 / 32 | 9,172 / 31 |
| 762020 | 533 / 4 | 738 / 7 | 370 / 3 | 447 / 4 |

614810 的 +60 图中 `(318,580)` 由 `[69,81,84]` 变为 `[121,141,147]`，左右旧暗带明显减轻。已查看左右 +60 和 face-neck +30。762020 已查看 face-neck ±30/±60、left +30/-60、right -30/+60，未见相对 matched0 的新增明显退化。

## 残线的可区分来源

614810 `(302,571)` 在原 write mask 外、输入 alpha=0，源和旧 guard4 输出均 `[106,126,132]`；向内 `(304,571)` alpha≈0.179，B 为 `[120,140,149]`。这说明单改原 mask 内 B 无法改掉源外被固定为背景的旧边缘。raw MEMatte 非真值，不能直接拿其非零值作为移动许可。

后续由 root 单独评估 trimap 外 4px 降级 unknown 的假设。本子实验仅提供同模型输入下此前被丢弃的外圈背景预测，供配对回放，不自行修改 alpha 或运动域。

## 同输入的外圈背景预测捕获

`guard_outer_capture.py` 使用原 mask 膨胀 4px 作为写入 target，`mask_dilation=0`。对不透明源和 full tile，这与原 target 加 `mask_dilation=4` 的模型输入相同；捕获结果为 worker 量化后的模型 RGB，并非浮点 raw tensor。

614810 沿用相同 23 个 tile 和 sin² 权重，外圈 7,158px 全覆盖。旧 mask 内逐 tile 结果仅 tile2 有 4px、最大 1 字节数值差，其余 22 个完全相同。最终场景严格保留之前 guard4 的整个原 mask 内结果，仅写新外圈；扩大域外 0 改动。所有外圈位置在覆盖它的 tile 中都有预测，未混入未预测的源像素。

- `(302,571)` 外圈 B：`[106,126,132] → [120,140,148]`。
- `(300,571)` 外圈 B：`[116,136,143] → [120,140,149]`。
- 原 descriptor 的外圈 scene weights 均为 1，无权重空洞。
- 762020 外圈捕获采用其已冻结的 24 个有效 tile，旧 mask 内逐 tile RGBA 全部完全相同，最终旧 mask 内与扩大域外 0 改动。外圈 7,489px 全覆盖；其中 9px 有部分相交 tile 未预测该位置，合成仅使用实际预测该点的 tile 的原 sin² 权重，未混入源像素。不新增参数选择。

## 已知真值的静止背景暗线反例

另以 `static_line_counterexample.py` 冻结一张 256² 合成图：有纹理椭圆前景、sigma1.25/truncate3 的软轮廓、独立静止竖暗线。原内外 6px unknown 包含全部真实软 alpha；将其外 4px hardBG 降为 unknown，各运行一次相同 MEMatte，无输出驱动调参。

新未知带 2,344px 的真实 alpha 全为 0，其中 190px 在暗线。结果：

| 指标 | 原 6px trimap | 外扩 4px unknown |
|---|---:|---:|
| 全图 alpha MAE | 0.001012 | 0.001094 |
| 新带 alpha 最大值 | 0 | 0.000429 |
| 暗线 alpha 最大值 | 0 | 0.0000471 |
| 新带 alpha >0.01 像素 | 0 | 0 |
| 假设左移 8px 后纯背景最大 RGB 误差（字节单位） | 0.00390 | 0.00467 |
| 上述误差 >1 字节像素 | 0 | 0 |

该简单反例未触发可见误拖；alpha 误差微增。模拟只说明“如果将所估计层整体平移”的风险，未授权 core 移动这些像素。合成图分布简单，一次通过不能证明真实阴影、毛发、复杂纹理的泛化安全，也不是人像审美验收。

## 证据与限制

### 762020 新 alpha + 外圈 B 的最终独立视觉复核

root 完成三臂普通 API 回放后，本子 agent 独立检查了左右边缘 ±30/±60、整图 ±30 对照、±60 单张原尺寸、四档原尺寸颈肩裁剪，以及最大差异处 ±60 三倍最近邻裁剪。相对 old-matte，未检出新的残线、光晕、耳廓缺损或肩颈断层，外发丝和软边连续；可判这张样例通过本轮新增退化检查，不能据此宣布所有功能可用。

new-pair 相对 old-matte 的四档像素变化为：+30 2,346px/max17；+60 2,993px/max30；-30 2,117px/max23；-60 2,847px/max52。最大差异在右颈/头发软边，例如 +60 `(622,451)` 与 -60 `(630,446)`；放大后是软边覆盖重新分配，未见额外硬边或孤立色块。肩衣所在 y≥500 区域四档仅 19/37/22/32px 变化，最大均 1 字节。零档恒等，oldOfficial 与 priorPaired 各档均 0 差异。

额外审查材料保存在该图 `optical-boundary-review-20261119/` 的 `independent-neck-shoulder-*.png`、`independent-right-neck-3x-*.png`、`independent-762020-diff.json`，未新增模型推理或修改算法。

### 用户原图防回归（同规则、无新检测）

使用用户原图 `test/results/user-face-contour-20261014/768/canonical-source.png`，SHA256 `0076be94ee7650c06bdf5f7850e7ae92e0f95305af790351e8f103df66739397`。核对 `user-head-neural-scene/report.json` 后确认它绑定 `user-head-observed-scale-full-source`，完整 768² 源 crop、trimap>0 与源补全 mask 一致。新目录 `test/results/face-generalization-20261118/user-original/` 仅以 symlink 适配历史布局，`adaptation.json` 记录绑定；没有重跑检测、更换源图或改几何/运动许可。

按同规则仅运行一次新 unknown MEMatte，以及 25 个固定 256/128 tile 的 guard4 扩展 target 捕获；接受原内 32px 的 65,882px 与外 4px 的 9,486px，覆盖无空洞，接受域外 scene 0 改。耗时为 MEMatte 3.05s、局部场景 40.07s，仅为本机一次前置准备。没有 guard0 模型重复推理。

普通 API 比较“原指定 paired-alpha/B”与“新 unknown alpha + guarded B”两臂各 0/±30/±60。原 baseline 四档与历史 48-photo 普通回放逐 RGBA 相同，两臂零档恒等。新旧四档差异 +30 4,317px/max25、+60 5,808px/max47、-30 3,210px/max28、-60 3,446px/max60。原 paired alpha 与原 official 已存在覆盖修正差异，此处仅是候选相对上一版的防回归，不是纯 alpha/B 因果分离。

已逐一看四档 face-neck、左右边缘，±60 原尺寸整图与左颈最大变化点三倍最近邻裁剪；未检出新裂缝、分离残线、耳饰断裂或肩颈色块。最大变化在左颈软边，+60 `(299,458)` 与 -60 `(295,456)`；局部保持连续。所有档 y≥590 对上一版 0 改动。可判该用户图通过本轮明显新增退化检查；这不重新验收既有脸型曲线，不代表所有强度/功能已达 8 分。结果在 `paired-regression-20261119/`。

- `test/results/face-generalization-20261118/<case>/context-guard-valid-20261119/`：冻结协议、tile、两组五档普通入口回放与视觉对照。
- `.../<case>/context-guard-outer-capture-20261119/`：量化预测、外圈覆盖、配对 descriptor、逐 tile 数值一致性。
- `test/results/optical-boundary-static-line-20261119/`：真值 alpha/B/F、原输入、两份 trimap、两次模型结果和风险示意。
- 背景预测不是隐藏背景真值；原 32px 接受带深边的拼接仍有潜在风险。两图/单一合成反例不能支持所有肤色、侧脸、逆光、遮挡与高强度的质量结论。
- 此记录不宣称整体可用或“8 分”。所有照片、模型和输出留在本地 ignored 目录。
