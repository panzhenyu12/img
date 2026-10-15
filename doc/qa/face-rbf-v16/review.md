# RBF/SVF + 可行侧轮廓目标 v16（隔离候选，未验收）

基于 v15，仅目标侧轮廓增加 Clarabel 凸二次规划。最小化请求偏差和位移弯曲能量，精确固定 attachment；目标切线与曲率不增加原轮廓之外的反向凸包/凹槽，允许原始像素轮廓噪声，所以源轮廓本身总是可行。Wendland 速度、f32 RK4、同一 inverse physical alpha 不变。QP 求解 f64，真实图像流与采样 f32。

用户同一原图 ±30/±60/±100 六档固定 ears/accessories/hair 0、9实际五官内部0、RGBA alpha0，纯收缩 scene 背景0。扩张新增覆盖 scene 是允许的。六档 stride2 112320 probes 无采样折叠，最小Jac范围 .362–.873，最大正反误差 .000367px。五项数学/几何测试通过。

±100 clear lower jaw 仍达到约20px请求量；earring/ear attachment rows 保持0；过渡段按可行边界递增移动。已导出 requested、feasible、advected 坐标及五次 shooting residual。扩张的 v15 圆鼓包明显改善。不是统一减小滑块强度。

仍拒绝总体效果：六档瘦脸均有脸/脖子交界黑三角、旧阴影/块状皮肤拼接；右耳根细黑点仍在。父任务材料 probe 证实这些黑三角像素直接取自 reconstruct 选择的 layer0(scene)，相邻 layer1(neck)为肤色；这能确认黑色来源，不能严格证明每个隐藏像素的真实材质应为 neck，因为没有 amodal 真值。颈层延拓与 LaMa 的后续负例见 [独立材料实验](../face-amodal-neck-20261014/review.md)。最终可见 skin 轮廓必须评估 transported face + 固定已见 neck + 推断遮挡 neck 的联合 silhouette，不能把 face alpha 边缘当整体外形。

当前 QP 的实验验证范围仅 faceWidth；垂直 chin/forehead 不能复用锁定v的侧曲线约束。下一隔离阶段修正 selection/protect 在目标/射击控制权重遗漏，并避免垂直参数被侧QP取消；其他9参数/多原片、WASM parity/性能都未验收。生产/public WASM 不变。
