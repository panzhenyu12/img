# 耳饰局部高分辨率因果实验

2026-10-15。固定原图、graph、自动 box/positive prompt 和 corrected mixed adapter，在 box 中心取2×、4× context crop。每个 crop 独立 encode 一次，combined-point/box 各解码一次；共4次 encode、8次 decode，四个 raw alpha 全部保留。

所有 official argmax 均为候选2，孔洞诊断均0，crop skin-core retained均0。局部推理只会减少不确定边缘，不会自动证明未知像素是皮肤；source skin support 仍为右81、左141，没有增加。

右耳组件2的4× combined-point：旧组件排除723像素（支持81、unknown642），局部排除581（支持81、unknown500）；亮部 alpha 最小值 `.960→.980`。这是当前唯一同时显示 unknown 减少和亮金属尾部改善的候选，已交给 source-audit 做固定 V36 inverse 合成。

左耳组件1的4× combined-point：排除 `799→531`、unknown `658→390`，但亮部最小 alpha `.953→.885`；combined-box 更差，暂不直接接入。局部实验也改变了上下文，不能把收益完全解释成分辨率。

完整输入与实际 tokens：[report.json](../../experiments/zim-accessory-crop-resolution-20261015/outputs/report.json)；32项逐候选审计：[audit-report.json](../../experiments/zim-accessory-crop-resolution-20261015/outputs/audit-report.json)。
