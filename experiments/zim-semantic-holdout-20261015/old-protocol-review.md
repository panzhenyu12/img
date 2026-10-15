# 旧 mixed 调用的独立样本：仅保留为失败证据

两张独立原照为 makeup（1280×1128）和 glasses-street（1280×853）。它们没有参与 woman/user/hair 三图上的 ZIM 选择规则开发。全部 6 组提示、24 张原始浮点 alpha、官方 score argmax 和固定门槛自动选择均保留。没有改阈值、人工选索引或重检测编辑图。

**调用缺陷已确认：这些记录中命名为 `box-positive[-skin-negative]` 的调用，并未把点标签送进 ONNX。** 固定 vendor 的 decoder 在 boxes 非空时覆盖先前准备好的 point_coords/point_labels，实际只传 box 角点标签 2/3；点坐标只改变 predictor 的 point attention。因此不能据此声称模型“不尊重负点”，也不能把此轮视为正确点框联合协议的验收。

| 原图/原请求 | 官方候选 | 自动结果 | 可信 skin 的 alpha>=.5 比例 | 可信 skin 平均 alpha |
|---|---:|---:|---:|---:|
| makeup 大 hair 组件，正点请求 | 2 | 2 | 0.53% | 0.0074 |
| makeup 大 hair 组件，负点请求 | 2 | 2 | 0.93% | 0.0651 |
| makeup 小 hair 组件，正点请求 | 2 | abstain | — | — |
| makeup 小 hair 组件，负点请求 | 2 | abstain | — | — |
| glasses，正点请求 | 2 | 2 | 2.10% | 0.0544 |
| glasses，负点请求 | 2 | 0 | 1.79% | 0.1419 |

此表不表示物理 alpha 精度。glasses 自动候选 0 通过旧二值门槛，但整张脸仍存在低透明幽影；77.99% 的可信 skin 像素 alpha>0.05。仅统计 alpha>=.5 会漏掉对物理合成有害的半透明残留。makeup 小组件在全部 hair 核心的保留率仅约 0.6%，所以全部拒绝；这是全局核心门槛应用于局部组件的已知适用性限制，本轮未改变规则。

饰品完整性也未被证实：makeup 原 parser 的 ears/accessories>=.9 均为 0；glasses 的 accessories>=.9 为 0，ears 核心 664 像素。因此这两张图不能提供独立的高置信饰品核心验收，空参考不应记为保护成功。原尺寸图中的眼镜/耳饰仍须人工视觉核对。

结论：自动筛选可以拒绝部分语义明显错误输出，但门槛通过不等于 physical matte 可用。旧 mixed 结果不用于生产、不用于 8 分声明。

证据：`raw/report.json`、`selection/report.json`、每组四 alpha、`selection/*/face-comparison-1x.png`。固定 `selector-snapshot.py` 保留原门槛。原始数据未更改，只新增本纠正说明。
