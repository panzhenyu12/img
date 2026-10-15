# 耳饰局部 2×/4× 分辨率与上下文实验

隔离实验，2026-10-15。输入固定为用户图 `canonical-source.png`、原始 graph 和既有两个自动 accessories 组件。每个组件以原 box 的中心为中心，取 box inclusive 尺寸的 2×、4× context crop；不手选坐标，不重跑检测，不改 alpha 阈值或生产代码。局部图独立 encoder 一次，然后执行 corrected `combined-point` 与 `combined-box` 两次 decoder；每次保留四个 raw alpha。

局部 alpha 通过“减 crop origin → 推理 → 原点粘回”映射回 1024×1024 原图。报告同时保存 local/full 坐标、crop 尺寸、实际 ONNX token/attention hash、全部候选、模型与源输入 hash，以及同区域 full-image baseline 的 alpha 差。

```sh
services/background-worker/.venv/bin/python experiments/zim-accessory-crop-resolution-20261015/run.py
services/background-worker/.venv/bin/python experiments/zim-accessory-crop-resolution-20261015/audit.py
```

本轮固定 B=1。2×和4×同时改变了有效输入分辨率与可见上下文，所以结果不能单独归因某一个因素。模型 CPU encode 每个 crop 约13–15秒，不能作为实时性能结论。

## 结果与是否值得合成

所有8次官方 argmax 均为 candidate 2；每个候选都保留。binary 封闭孔洞均为0，source-supported skin 数量没有增加，crop skin-core retained 均为0。局部结果不是物理饰品 matte。

| 组件 / 模式 | crop | 排除旧组件像素 | 源皮肤支持 | unknown | 亮部 alpha 均值 / 最小值 | 同 crop 对 full alpha 平均差 |
|---|---|---:|---:|---:|---:|---:|
| 右2 full combined-point | — | 723 | 81 | 642 | .99963 / .960 | — |
| 右2 4× combined-point | [558,309,754,621] | 581 | 81 | 500 | .99947 / .980 | .00244 |
| 右2 4× combined-box | [558,309,754,621] | 640 | 81 | 559 | .99925 / .971 | .00234 |
| 左1 full combined-point | — | 799 | 141 | 658 | .99934 / .953 | — |
| 左1 4× combined-point | [266,301,458,621] | 531 | 141 | 390 | .99842 / .885 | .00220 |
| 左1 4× combined-box | [266,301,458,621] | 799 | 141 | 658 | .99726 / .778 | .00148 |

右耳饰组件2的4× combined-point 同时减少 unknown 边缘并提高亮部最低 alpha，具备送入固定 V36 inverse 的候选价值。左组件1的4×结果虽减少 unknown，但金属亮部最低 alpha 下降，暂不直接替换；必须经过 source/F-B 合成检查。2×和 box attention 没有形成两项同时改善的证据。

`audit-report.json` 中还包含全部32个 candidate 的同域指标；这张表只列 official candidate 2。没有根据缩略图挑选候选，也没有宣称达到8分。
