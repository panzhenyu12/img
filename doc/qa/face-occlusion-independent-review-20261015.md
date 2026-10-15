# 头发与皮肤遮挡 A/B 独立复核

2026-10-15。完整瘦脸候选仍不通过画质验收，不支持 8 分或上线；本审计未修改生产算法。

独立检查三张原照 ±100，加 woman +100 的三种 LaMa backing，共 9 组/18 张输出。固定头发纹理坐标不等于固定其可见性：B 允许脸遮住后层头发，hair 扩脸时可明显减少黑条；前后层关系仍未知，不能自动抹去真实前景发丝。woman 收脸仍有硬边/补全带，hair 收脸仍有锯齿与颈部连接问题，用户原照两方向完全 no-op。

新增合成步骤的写域、ROI、人物覆盖、耳饰区域及 alpha 合同通过，但 V36 继承的耳饰误差仍在。独立反例发现 unknown 区域可通过 transported edge 绕过禁写；已由实验作者修复，6 个违例写入降至 0，前后 90 个真实 PNG 字节相同。失败脚本与反例完整保留，未放宽旧回归。

ZIM 新 mixed adapter 的点/框 token 接入符合实际 feed；3 项合同通过，6 组旧 mixed alpha 独立验证与历史数组逐浮点相同。新 A/B 实验确实使用修正 prompt 的 alpha；旧实验结果不得因此被追溯改写。

详细审计：[review.md](../../experiments/hair-skin-decomposition-v1-20261015/independent-occlusion-audit/review.md)。独立数值：[report.json](../../experiments/hair-skin-decomposition-v1-20261015/independent-occlusion-audit/report.json)。红/绿反例：[unknown-counterexample.json](../../experiments/hair-skin-decomposition-v1-20261015/independent-occlusion-audit/unknown-counterexample.json)。
