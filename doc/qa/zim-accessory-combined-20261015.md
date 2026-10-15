# ZIM 耳饰提示与源肤色条审计

2026-10-15。复用用户图既有 1024px canonical source/graph 与两个自动耳饰组件，保留五种调用模式、每组四个原始候选及实际 ONNX token/attention 证据。只验证单图 B=1；没有重新画 mask、降低强度、改几何或改生产代码。

上游 mixed 调用会用 box token 覆盖正点；corrected adapter 将正点与 box token 一起传入。两个组件的十组官方 argmax 均为候选 2，但官方分数不代表语义质量。

原分割把耳饰旁的脸颊肤色条包进了固定区域。combined-point 共排除 1522 个原饰品参考像素，其中仅 222 个有独立源皮肤/Lab 近邻证据（图像右81、左141）；其它1300个保持 unknown。孔洞、深色背景和反射不因为饰品 alpha 低而变成皮肤。

旧 mixed 在这两组件也得到相同222个支持像素；本轮 corrected adapter 没有新增皮肤召回，不能把输入契约修正当作已证实的画质提升。

原尺寸检查显示亮金属链条基本保留；源高亮尾部 alpha 均值约 .9996 / .9993，但该指标不是金属真值，也不是固定纹理保真证明。原始候选的二值内部孔洞为零，不足以证明实物没有穿孔。耳钉连接和混色边还需在固定饰品前层、皮肤后层的 F/B 实验中验证。

本轮是源图候选审计，未通过实际 ±100 变形与物理边缘验收，不声明达到8分。

[完整评审与五模式对照](../../experiments/zim-accessory-combined-20261015/review.md) · [实际模型输入与全部候选](../../experiments/zim-accessory-combined-20261015/outputs/report.json) · [源证据诊断](../../experiments/zim-accessory-combined-20261015/outputs/audit-report.json)
