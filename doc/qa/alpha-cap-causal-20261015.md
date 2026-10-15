# Alpha 上限因果审核

2026-10-15。固定三张已有原图的 raw ZIM alpha、解析器核心、donor、solve domain、inverse map 与 A/B 合成，只改变 source-fit alpha 规则。没有重新推理、重画 mask、改几何或改生产代码。

当前 `±0.15` 版本可以逐字节复现原 `source-fit-srgb` 层与既有 A/B 输出。无上限版本在 woman mixed 域 3,620 像素、portrait-hair 437 像素触发；源图 RGB 重建误差下降，但 woman +100 的旧橙色细线仍存在。部分残线已由独立实验证明来自错误 H/backing 颜色，不能将全部残线归因 alpha 上限。

已知颜色与 alpha 合成反例更关键：真细发 alpha=.65，但细发颜色接近肤色与不透明头发 donor 的混合。无上限拟合把真 alpha 降到约 .364，源重建误差从 0.682→0.011 byte，却让已知新背景误差 33.87→43.67 byte。局部 donor-variance 区间同样受系统性错误 donor 误导；物理无解区间现在 abstain，不把两端裁切成伪解。

因此不能接受“取消 cap”作为修复。当前 cap 仍是安全边界；需要独立前景/背景 donor、视频多帧或额外物理约束后再研究 alpha 更新。`verification.json` 的 `algorithmAccepted` 明确为 `false`。

[实验说明](../../experiments/alpha-cap-causal-20261015/README.md) · [完整 source/A-B/synthetic reports](../../experiments/alpha-cap-causal-20261015/verification.json)
