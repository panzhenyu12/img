# ZIM 联合提示与自动选择：三张独立原照验证

日期：2026-10-15。新增 makeup 1280×1128、glasses-street 1280×853、dark 1280×1920 三张独立原照，不把 roll 或缩小版本计为独立样本。未修改生产、vendor 或旧测试记录。

**确认并修正实验调用层的关键问题：旧点+框调用没有将点标签传入 ONNX。** `decoder.py` 的 boxes 分支覆盖 point tokens；点坐标仅影响 attention。因此旧“负点没有起作用”的现象不能直接归因于模型能力。独立捕获验证旧 wrapper 翻转正负标签后实际输入完全相同；正确拼接后，真实 ONNX 的输出随标签改变。动态4/7 token均可运行。

root 的隔离 adapter 经过 B=1 评审。holdout 运行器在加载前校验固定权重 SHA，并用透明 recording proxy 捕获 50 组真实 ORT feeds；五模式分别为 upstream-mixed、box-only、points-only、combined-point、combined-box。每个模式保留四个原始浮点 alpha、官方 argmax 和固定旧门槛的自动选择。每图只编码一次，复用 features。

| 主 hair 组件，正+负请求 | 旧 mixed 自动候选的可信 skin 平均 alpha | 正确 combined-point | 该新候选 hair 核心保留 |
|---|---:|---:|---:|
| makeup | .0651 | .0040 | 91.68% |
| glasses-street | .1419 | .0070 | 91.45% |
| dark | .0065 | .0091 | 85.23% |

makeup/glasses 的整脸低透明幽影明显减轻，证明正确传递标签有实效；dark 添加负点后，主组件 hair 核心保留从正点协议95.36%降到85.23%，右侧卷发出现损失/半透明，必须保留为失败。小发束在全局 hair-core 门槛下仍会 abstain，没有放宽阈值提高通过数。50组中32组有合格候选、18组abstain；这不是matting精度或通过率评分。

三图 accessories>=.9参考核心都为0，尽管图中能看到眼镜或耳饰；makeup ears核心也为0。glasses/dark可用ear核心分别664/1718，新负点hair候选在这些核心内alpha接近0。**空饰品参考不等于饰品已保护。** 原尺寸视觉和额外 soft-alpha审计保留，不能仅靠alpha>=.5门槛宣布物理matte可用。

结论：可以把修正后的调用作为后续 source physical matte 的候选输入路径；不能直接无条件替换现有层归属，不能把 non-hair 自动并入 skin。需要继续验证发丝保留、半透明边缘颜色分解、饰品保护与变形后的实际合成。本轮没有执行新 warp、没有上线、没有8分结论。

- [完整实验与复现](../../experiments/zim-semantic-holdout-20261015/README.md)
- [旧协议纠正与负例](../../experiments/zim-semantic-holdout-20261015/old-protocol-review.md)
- [五模式审计](../../experiments/zim-semantic-holdout-20261015/combined-review.md)
- [50组真实 feeds](../../experiments/zim-semantic-holdout-20261015/combined-raw/report-with-feeds.json)
- [固定选择规则及逐部位审计](../../experiments/zim-semantic-holdout-20261015/combined-selection/report.json)
