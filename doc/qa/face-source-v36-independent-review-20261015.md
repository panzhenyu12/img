# V36 独立复核：恢复运动，画质仍不合格

V36 取消了垂落头发对皮肤速度场的不合理固定，woman、hair、makeup 原图上的真实轮廓位移明显恢复。但 woman +100 留下旧亮肤色轮廓和无纹理棕条；makeup +100 在耳下出现硬阶梯，−100 出现嘴角暗线延伸。**拒绝上线，不能认定达到 8 分。**

独立审计覆盖 19 组原图与强度。19 份操作图与 V35 一致，16 个非零案例的源物理 alpha 与 V35 逐字节一致；用 V35 固定二进制回放同一 woman +100 输入，输出与既有 V35 的 SHA-256 完全相同。恢复运动的因果来源是 hair motion attachment 修改，检测与 mask 未被更换。

全量 Rust 回归为 207 个单测通过、3 个忽略、10 个公开 graph 合约通过，没有放宽断言。所有案例中，hair/ears/accessories **置信度 ≥0.5** 的固定像素变更数为 0，输出 alpha 变更数为 0；软边缘像素仍可变化。测试通过只证明这些约束，不能证明边缘真实自然。

原尺寸采样确认：woman 的三处旧亮边像素被 hair 误归属，以 0.878–0.960 的置信度原样固定。因此，“固定头发像素通过”本身也可能保留错误轮廓。该归属问题与缺少头发纹理的棕色补全条带需要分别解决。

完整审计、输入/输出哈希、原尺寸对照和失败记录位于 [V36 review](../../experiments/face-source-curve-v36-20261015/qa/review.md)，机器数据位于 [audit-report.json](../../experiments/face-source-curve-v36-20261015/qa/audit-report.json) 和 [summary.json](../../experiments/face-source-curve-v36-20261015/qa/summary.json)。本次审计没有修改候选或生产算法源文件。
