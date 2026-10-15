# Direct skin + accessory 联合支持的源背景（2026-10-15）

有效交付是 **`outputs-v2-consistent/scene-support.npz`**。`outputs/` 为第一次实验，旧 scene seed 自身包含 84 个与 direct skin/accessory 冲突的像素，已拒绝；脚本快照保留在 `outputs/run-snapshot.py`。不能把旧 scene seed 当真值永久保留。

本实验不修改图片、不移动皮肤、不修改 inverse/alpha，也不接生产。只为后续分层重建提供同侧、可追溯的**源图中已观测背景**和局部先验。

## 新证据

输入固定为用户 1024×1024 原图、既有 face graph，direct skin 使用 `joint-skin-matting-v1-20261015/outputs/{full,crop}-point/alphas-full.npy` 官方候选 index 2；corrected accessory 使用已审计的 full combined-point alpha。

局部场景候选必须同时满足 full skin < .01、crop skin < .01、accessory alpha < .01、旧 hair/ears/五官 < .05。以旧 strict scene 与上述证据一致的像素为 seed，候选还必须与 seed 四连通，路径每步源图 Lab 色差 ≤8、亮度差 ≤6、色度差 ≤6。没有把“模型排除”自动解释成背景，也不允许跨金属去填封闭孔洞。

| 项目 | 左耳饰 | 右耳饰 |
|---|---:|---:|
| 新增 observed scene | 762 px | 735 px |
| 新增域中旧 accessory≥.05 | 511 px | 453 px |
| 旧 seed 与新证据冲突而剔除 | 84 px | 0 |
| 颜色路径拒绝 | 7 px | 5 px |
| 无法连接真实 scene 而拒绝 | 0 | 97 px |
| source skin support 到旧 scene 中位距离 | 28.12 px | 31.02 px |
| source skin support 到新 scene 中位距离 | 6.71 px | 21.00 px |
| direct mixed band 到新 scene 中位距离 | 2.00 px | 2.24 px |

新增 1,497 个 scene 像素与旧 pure/trusted skin support 重叠为 0；protected foreground 重叠为 0。原尺寸 overlay 可见新增域主要为耳饰与脸之间原本可见的深色背景。距离明显改善证明旧 parser 的过度冻结确实让 donor 被迫远离目标。右侧部分源支持仍离可靠背景较远，不能声称隐藏背景真实可知。

## 数据合同

`outputs-v2-consistent/scene-support.npz` 为 1024×1024 坐标：

- `scenePrior`：float32 RGB，0..1。由同侧已观测 scene 的 5×5 有效均值及最近点延拓得到；**隐藏部分只是颜色先验，不是真值**。
- `sceneDistance`：float32，原图像素单位，仅两个 context 内有效；**context 外为 inf，调用方必须用 finite gate**。不可把 context 外的 `scenePrior` 原图值当 scene donor。
- `nearest`：int32，`[2,H,W]`，顺序 y/x；仅 finite distance 域有效，目标与 donor 在同一个侧的 context 内。
- `addedScene`：新增的直接可观测支持。
- `strictScene`：已与 direct masks 对齐的 seed。
- `legacyStrictScene`：旧 strict scene，仅用于对照。
- `newScene`：consistent seed + added scene。

两个 context 分别是 `[306,389,417,532]`、`[600,394,712,535]`。每个 component 的 `scene.npz` 额外含 region、oldPrior/oldDistance，用于对比。`source-evidence-1x.png` 是原尺寸三列证据：原图、新支持 cyan/保护 green、scene prior（最后一列是先验图，**不是成图候选**）。

## 检查

```sh
services/background-worker/.venv/bin/python experiments/joint-skin-scene-support-v1-20261015/test_scene_support.py
```

检查 source/graph/input/output SHA、所有 scene 的前景排除、同侧 nearest 与距离一致、RGB 四邻接图连通、已观测 scene RGB 保真、context 外 inf，以及旧 seed 冲突确实被剔除。`acceptedForProduction=false`，最终可用性仍须由后续 ±100 实际成图决定。
