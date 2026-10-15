# Wendland RBF + SVF v14（未接生产）

private source-forward Wendland C2 基函数；nalgebra Cholesky 解正则矩阵；f32 RK4 正反积分；5 次实际 advected boundary 残差拟合。不是从旧 MLS/nearest-segment 位移构造速度。feature/fixed/selection 约束进入速度，alpha 由同一 inverse flow 采样源 physical coverage。

+100 的 active boundary RMS 1.482→0.323px，minJac=0.501、-100=0.410；固定 ears/accessories/hair 0、alpha 0。三项数学单测通过。耳根大裂缝减轻，但下颌凹陷、脖子材质接缝仍失败。拒绝。数值指标不代表达到 8 分。

发现右下颌 tiny frozen `other_person` 岛：输入 subjectMask 是本人 parseFace 的 personAlpha 类并集，不是独立人物 instance。冻结了本人 face/neck 类低置信边界，引起实际速度在 y535 处归零。overlay 中 red=explicit fixed ears/accessories/hair，cyan=inferred non-face selected person。下一次只改这个误类，保留相同速度场/目标/颈部处理以进行因果消融。
