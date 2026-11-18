# FUNC-040 完整颈部观测域与独立材料约束

状态：**隔离算法候选，未默认启用 VectorNeck，未接入 Web/MCP。**

2026-10-29 保护域增量：独立复核发现原完整源观察器仍把 `protectMask` 当作材料排除，真实 caller 又把受编辑保护影响的 `joint_owned` 传给 VectorNeck。原回归中仅保护下半颈，就让观测底部从 320 截成 250。现改为源材料归属独立观察、按同源坐标供给向量颈域，保护仍进入速度硬零与输出权限。5 项相关回归通过，其中真实 `adjust` 调用验证保护前后 neck/low 身份相同、受保护区域原样且脸部仍有效形变。该增量未默认接入；无保护实图中和旧 RGB fringe ownership 的一致性仍需组合复核。此前“selection 不改变颈域”只在独立 API 上成立，不能覆盖这个已复现的 caller 错误。

## 解决的问题

原候选从已经裁剪的渲染网格计算 neck geodesic 长度，再固定末端 25%。同一张源图只要扩大计算 crop，就可能改变既有颈部像素的运动。只从全图计算 low mask、仍在截断网格解 harmonic，并不能消除新的边界条件。

另一个已复现的问题是明确衣物与 `0 < neck < 0.5` 重叠：衣物虽然速度被 fixed 固定，旧 soft 扩展仍把它纳入 neck 部位，污染观测范围和连接关系。

## 候选行为

1. 在完整源图的像素中心采样既有 faceContourMask、skin/neck 和明确材质证据，选择同一 face-connected 观察支持。保留原始 mask 和解析分辨率；不把 crop 外证据补零。
2. 根据完整组件 bbox、sampling reach 和额外零边界 halo 确定 canonical source ROI，同时保留原有 head/context 支持。全部 ownership、perimeter、harmonic 和 RK4 都在完整域内求解。
3. 显示或导出窗口仅读取已定义的源场，不能作为新的部位边界，也不能在积分前截断场。
4. 独立 `material_blocked` 排除确认衣物和真实前置 hair/ear/accessory。运动约束 `fixed` 仍可包含 user selection 或 region；后者固定真正的颈部，但不能借此缩短其观测长度。

接口保留旧无材料入口，并新增 `VectorNeck::build_with_material_boundary`。新入口对 face/core/soft 使用同一明确材料排除；确认衣物不再被当作 foreign neck。完整源观察适配器采用相同材料规则。

这是当前语义观察支持的定义，不证明分割出的边界都是真实解剖边界；原末端 25% 仍是显式工程先验。

## 验证与结果

- 不含图片和模型的独立测试入口：`bash experiments/face-neck-source-domain-20261028/run-tests.sh`，20 项测试通过。
- 保留旧裁剪错误复现：crop 高60时 source y50 被固定；完整高100时该点仍有 vx > 3 的运动。完整域满足矩形解析解，未靠冻结整条颈取得不变性。
- 明确材料桥完成红→绿：相同衣物/soft-neck 输入，旧入口错误纳入，新显式入口排除；其余真实颈部仍运动，face Dirichlet 幅度保持。
- movement selection 反例：固定下半颈时，neck/low 定义不变，只有所选区域速度为零。
- 隔离 caller 的 37 项颈部定向测试通过；完整域与材料修复合并完成 8 张真图回放，未重复模型推理。

真图组包含 dark ±60/±100、hair ±60、tilted ±60。dark 的旧计算域遗漏 1965 个 neck 支持像素；补全后仍保持前128个 controls/requested/feasible 不变，衣物与 low 固定像素无改动。各组离散相邻 Jacobian 为正，所有活动源点的双向 RK4 子阶段未出计算域。这些是所测样本的证据，不是连续全域的数学保证。

## 未完成的边界

dark 完整域再加16px halo，几何场、low、alpha、frozen 和目标逐元素一致，但源背景 B 仍有变化，因此最终照片还不是 padding 不变。需要继续分离源光学重建的工作域依赖。

这是本候选冻结包的光学限制。独立的 [FUNC-039](FUNC-039-source-optical-domain.md) 已在旧完整颈部组合上验证 halo 增加16px后最终输出逐字节相同；尚未与本候选的新材料桥或后续响应修复合并回放，不能将该结果直接外推到新组合。

tilted −60 控制69仍只有 feasible 位移的 68.2238%（.7237px → .4937px）。完整颈部域没有修复该局部低响应；不能用全脸总响应掩盖它。

目前材料输入需要显式提供，自动检测的泛化、公共协议校验、浏览器集成与整体自然度验收均不属于本候选已完成内容。不得以这项修复宣称全功能可用或达到 8 分。

独立 API 的 movement selection 测试最初漏掉了 caller 中的 `protectMask` 截断。该错误已由 2026-10-29 的源观察与真实 `adjust` 回归修正，见本页增量记录；真实多图组合仍须单独核对，不能以一个保护 fixture 证明全部复杂遮挡可靠。

源码、独立测试和接入参考仅保存在 `experiments/face-neck-source-domain-20261028/`；未复制完整 core 或提交照片、mask、模型、trace。主 `face.rs` 未由本工作修改。
