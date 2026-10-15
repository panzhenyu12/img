# 整眼刚性 / 邻接皮肤延拓单一研究方向

生产 geometry.rs 保持 V4b。首版“扫掠polygon核心 + 距离smoothstep外场”已失败归档，见 [first-rejected/review.md](first-rejected/review.md)，不得推广到正式Web/API。

后续仍只改变 eyeTilt 和 eyeSpacing。完整可见眼缘、虹膜及估计的上下眼睑/眼袋邻接包络遵循同一SE(2)generator，外部皮肤采用局部screened biharmonic速度延拓。固定眉/鼻梁、对侧眼和用户保护是独立边界；同时编辑眉时，用眼的整个运动扫掠域保护，不只静态eye bounds。

目标曲线从原眼polygon出发。邻接包络是现有检测加邻接估计，不是独立检测了眼睑折线；原尺寸照片必须判定它是否真正带走了双眼皮和眼袋纹理。运动预算冲突时沿该请求轨迹求可行姿态，并连续饱和到首个约束，不突然关闭整项。QA分别记录requested/applied，并验证临界值两侧。

数值验收至少包括：完整眼polygon边长/面积、actual final inverse mapper回原点、内部iris-sized circle（仅几何代理）、V4b完整mapper/frozen/跨零/rotation回归、眉眼组合，以及原尺寸眉到眼下皮肤比较。成功的数值不能替代外部皮肤自然度。

先复核实际失败用户照片±100；如果仍有眼睑/鼻梁捏痕，停止扩大照片矩阵并归档失败。仅在这一步通过之后扩大为其他独立原照±30/±100。
