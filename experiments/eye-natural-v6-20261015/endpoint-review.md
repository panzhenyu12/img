# 端点映射与膜应变对照：均拒绝部署

裸端点版：将刚性眼的目标位置作为约束，在同一双眼域求有限 SE(2) 位移，直接作为逆映射；逆向采用数值求解。用户原尺寸 0/±100 与源图一致检测。默认 geometry 为 31 pass / 1 fail / 3 ignored，eyeSpacing +100 有 15 个负 Jacobian、最小 -0.1697165、逆误差 3.4118px。失败档位的最大位移 4.7388px 对应 256px fixture 请求，并非有证据的幅度超调。

加入 0.5 倍离散膜能量（梯度 L 项）后：原 geometry 32 pass / 0 fail / 3 ignored，完整用户图 0/±30/±100 已复放；整眼姿态仍按完整请求，整眼误差小。**独立视觉审查仍拒绝**：eyeSpacing -100 的鼻根被压细，比 V4b 更明显；+100 鼻梁仍有方形明暗过渡。全绿只覆盖原测试，不构成算法验收。

补充新的眼编辑选区/保护边界回归后，再次失败：eyeSpacing +100 + 硬保护区出现 12 个折叠，最小 Jacobian -0.1452073、逆误差 6.1807px。旧 86 场景的边界部分仅测眉编辑；新测试已保留在默认源码中，没有删改失败条件。

资料：`endpoint-source` / `membrane-endpoint-source`，`endpoint-qa` / `membrane-qa`，`geometry-tests-endpoint.log` / `geometry-tests-membrane.log` / `eye-boundary-regression.log`，不可变 worker 的 SHA 文件。

后续可行性实验回到稳定 SVF。约束眼内角与源鼻根/鼻梁/眉之间的最小皮肤空间，温和档保留请求；强档明确给出请求、可行幅度、实际幅度及限制部位，并与 V4b 匹配实际位移的结果比较。减小强度本身不得算作画质提升。
