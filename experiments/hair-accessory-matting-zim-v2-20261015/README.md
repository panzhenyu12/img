# 原图 hair/accessory 物理边缘候选：ZIM 自动提示审计

**结论：有局部边缘分解收益，但还不能解决 V36 白线，也不能通用替换 hair 分割。**

**后续接口审计更正：**官方 `vendor/zim_anything/modeling/decoder.py` 在 `boxes is not None` 时，用 box 坐标/标签覆盖传入的 point 坐标/标签；predictor 仍使用 points 建立 attention mask。因此本文历史 `box + point` 实验实际不是同时生效的 box/点标签提示。尤其“加负点”结果不能解释成模型不遵守负点，其影响经过 attention mask 路径。以下 raw 结果/名称/计数完整保留；真正混合 box + signed point 的 adapter 对照由独立实验补充。颜色归属的失败结论仍成立，但提示机制的因果解释以本更正为准。

固定官方 ICCV 2025 ZIM 源码和已有权重，不重新下载，不改 production、Rust、WASM、服务或 graph。三张原照分别是 `portrait-woman`（1280×1920）、`user-face-contour`（1024×1024）与 `portrait-hair`（588×768）。实际执行了原图推理，没有对编辑结果重新检测，没有图片专属提示窗口，没有人工挑选低分候选。所有来源、模型与图像 hash 在 `outputs/report.json`。

## 固定自动协议

复用 accessory-v1 的 `confidence >= .5` 连通组件、面积 >30、最多两个最大组件、框外扩 4px、组件内部距离变换最大点。饰品沿用组件中心在下巴以上的过滤；长发允许中心在下巴以下，但组件须延伸到下巴以上。

两个协议均保留四个 raw float alpha、四张 PNG 与各自官方 score argmax；不按视觉结果在候选之间切换：

1. `component-box-positive`：连通组件框 + 一个内部正点。
2. `component-box-positive-skin-negative`：在同一框内增加一个由原 skin>=.85、hair<.05 内部距离最大值生成的皮肤负点。只有 hair 使用此对照；饰品继续使用协议 1。

## 实测结果

下表的皮肤误含只是和冻结原图可信 skin 核心比较，不是 alpha 真值精度。所有 official argmax 都是索引 2。

| 原图 | 协议 | 原 hair 核心保留 | 可信 skin 核心误含 | 归属候选 |
|---|---|---:|---:|---|
| woman | 框 + 正点 | 87.71% | 0.66% | 187px，均来自旧 hair |
| woman | 再加自动 skin 负点 | 88.53% | 97.19% | 拒绝；负点 alpha=.778 |
| user | 框 + 正点 | 96.93% | 1.68% | 222px，均来自旧 accessory |
| user | 再加自动 skin 负点 | 96.91% | 1.65% | 同样 222px accessory |
| hair | 框 + 正点 | 97.52% | 73.21% | 拒绝，误选整张脸 |
| hair | 再加自动 skin 负点 | 97.80% | 77.62% | 拒绝；负点 alpha=.995 |

woman 的该自动协议确实降低了旧 hair 中的左颊肤色细线；user 的饰品结果也复现了旧饰品 mask 包含皮肤带/孔隙背景的问题。**此接口路径下添加点的对照没有稳定改善；不能把较高的官方 score 当作“hair 语义正确”，也不能据此评价真正的负点标签遵从性。** `portrait-hair` 的低分候选看起来有更强的排脸效果，但本次没有替换 official argmax，不计为成功。

CPU 3 线程：模型初始化 1.78s，三张原图编码分别 7.15/7.73/6.42s；单个提示的解码及后处理约 0.44–0.99s。这不是 WASM/WebGPU 或滑杆实时性能。

## 严格的原图归属候选

`analyze.py` 只生成候选 PNG，不把 raw ZIM 全脸 union 到 skin，不使用 mask dilation，不产生可直接发布的 graph。

像素必须同时满足：

- 原 hair/accessories>=.5，所对应的 ZIM alpha<=.05，而且新的 hair 与 accessory 都<=.05；
- 原 skin>=.5，或与最近的局部可信 skin 颜色均值满足 ΔEab<=8、亮度差<=6、色度差<=6；
- 到可信 skin 的距离<=`min(3 个原检测格点, .025×landmark 脸宽)`，只作为证据距离否决条件；
- 不在耳朵、眼睛、眉毛、嘴唇、牙齿保护区域内，并处于自动 landmark 取出的脸部范围。

模型如果明显保留皮肤负点、丢失正点、误含超过 5% 可信 skin，或者保留不到 65% 原 hair 核心，则该协议不生成任何归属修改。通过这些检查仅表示未被这组规则否决，不代表物理 alpha 已正确。模型排除的其余像素包括背景、饰品空隙、暗发丝及不确定边缘；它们不会自动改为皮肤。

## V36 独立白线探针没有通过

其他审计先前记录的坐标只用于事后评估，完全不进入 prompt、alpha、颜色门限或归属规则。

| 原图点 | 旧 hair | ZIM official hair alpha | 与邻近可信 skin 的 ΔEab | 归回皮肤 |
|---|---:|---:|---:|---|
| (395,960) | .9603 | .2019 | 19.55 | 否 |
| (418,1000) | .9374 | .0625 | 9.77 | 否 |
| (453,1040) | .8780 | .0719 | 11.90 | 否 |

这些点仍存在模型软 alpha 和明显变暗的原图边缘，不能将其当作纯皮肤。它可能涉及源照片的混色边缘，也可能包含 ZIM alpha 误差；本审计没有前景/背景颜色真值，不能据此断言物理混合比例。直接放宽阈值并移动这些旧 RGB，仍可能携带暗边；直接冻结也会保留旧轮廓。**下一步需要专门验证边缘 alpha/颜色分解和变形后的遮挡合成，而不是靠宽泛扩张 skin 结束问题。** 当前只完成源边缘诊断，没有执行新候选 warp，不声称白线消失或达到 8 分。

## 复现与结果

```sh
services/background-worker/.venv/bin/python experiments/hair-accessory-matting-zim-v2-20261015/run.py
services/background-worker/.venv/bin/python experiments/hair-accessory-matting-zim-v2-20261015/analyze.py
services/background-worker/.venv/bin/python experiments/hair-accessory-matting-zim-v2-20261015/probe_observations.py
```

推理与归属输出目录必须不存在；不要覆盖已有 raw 证据。`run.py --out PATH` 可单独生成另一份原始推理；`analyze.py` 默认处理此目录下的 `outputs`。

- `outputs/report.json`：原始模型输入/输出、每个四候选的 score 和 alpha 指标、CPU耗时。
- `outputs/ownership-report.json`：所有接受/拒绝理由、归属候选规则和连通区域。
- `outputs/portrait-woman/ownership-component-box-positive/left-edge-1x.png`：原图、旧 hair、新 alpha cutout、绿=source 支持候选/粉=未知，四列均为原尺寸。
- 每张原图的 `ownership-*/face-raw-alpha-1x.png`：原图、旧灰度 alpha、新灰度 alpha，保持 1:1 像素。
- `outputs/user-face-contour/accessories-*/component-short-label-comparison-1x.png`：饰品四个原始 alpha 的局部原尺寸对比，`*` 是官方 argmax。
- `outputs/portrait-woman/independent-v36-probe-observations.json`：全部独立探针与四候选 alpha、颜色/距离观测。

`probe_observations.py` 同时导出灰度 alpha 对比图；原始推理与归属计算的脚本 hash 分别保留在对应 report 中。
