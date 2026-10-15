# eyeSpacing -30 / +30 同源对照

沿第三候选的单项机制继续验证同一张 1024×1024 用户原图：`abs(eyeSpacing) <= 30` 使用鼻梁有限软锚定 penalty 0.18；其它场景不变。两个方向都保留完整眼睛 SE(2) 刚性运动，不以缩小眼睛主体换取平滑。

- `eyeSpacing=-30`：左/右眼实际平移由诊断 JSON 记录，眼缘面积与边长比均为 1.0，最终 inverse rigid error 最大约 0.00102 px。
- `eyeSpacing=+30`：左眼 **+5.678 px**、右眼 **−5.686 px**，眼缘面积与边长比均为 1.0，最终 inverse rigid error 最大约 0.00128 px。

两档候选图在 `*-comparison-2x.png`；`comparison-report.json` 保存 source/worker hash、实际平移和相对第二候选的 RGB 差异。当前只做同源 ±30，尚未扩展 ±100。

这是算法机制验证，仍需人工确认鼻梁、内眼角和双眼皮过渡是否自然；任何一档出现新折痕都不能进入 production。
