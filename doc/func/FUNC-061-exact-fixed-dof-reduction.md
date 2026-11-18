# FUNC-061 有界形变场的精确固定自由度消元

日期：2026-11-13。状态：当前原生隔离求解器可选能力，已完成数值等价测试与三组真实规模配对；未接入 Web/WASM 默认流程。

## 问题与行为

原有界位移求解器将全部网格节点的 x/y 位移以及柔性刚体的共享平移作为未知数，再用等式固定背景、边框和刚体组。大量已知常量与重复自由度增加求解时间。

本功能对完整已装配二次规划进行精确代换 `x=Tz+c`：已固定或硬刚体的位移进入常量 c；未固定柔性刚体连通分量共享两维平移；其他自由节点保留两维。相同目标、保护、权重和可行域下，减少进入求解器的变量与恒成立的常量约束。

这项优化改善求解效率，不改善脸型、语义判断或边缘纹理。与 [FUNC-060](FUNC-060-contour-residual-coupling.md) 新增轮廓能量/投影边界不同，精确消元本身不增加或放松运动限制。

## 接口与实现

- `core/image-core/src/face_bounded_field.rs`：原 `solve(problem)` 仍走 baseline；新增 `solve_with_reduction(problem)`，内部共用 `solve_impl(problem, bool)`。不在 Problem 中新增开关，不改变 Solution 输出 schema。
- `core/image-core/src/face_bounded_reduction.rs`：std-only 通用变换。`from_fixed_shared` 验证已完成的刚体连通合并与固定传播；`reduce_qp` 接收原上三角 P、q、A triplets、b 与等式行数；`expand` 恢复原变量布局。
- 代换放在全部目标、轮廓残差、共旋材料、固定点、背景线和梯度约束装配之后，所以新二次项不需要另写一套 reduced 版本。

目标变为 `Pz=TᵀPT`、`qz=Tᵀ(q+Pc)`；同时保留常量以便独立检查原完整能量。约束变为 `ATz<=b-Ac` 或对应等式。映射后的同列系数相加，仅删除精确抵消的零行；常量矛盾必须报错。

关键约束：非零固定值的 `Pc` 不得遗漏；原非对角 Hessian 两变量合并时，两份对称贡献都进入新对角；共享刚体内每个原节点的正则仍全部计入。不能通过只计一次组正则改变问题。

恢复全变量后仍执行原固定/刚体、固定采样点、背景线、材料梯度、单元 Jacobian、边界及 targets 检查。原求解器 1e-10 可行性/gap 容差、梯度安全余量均不变。零自由度时先检查常量约束，再直接返回唯一常量解。

该 helper 只适用当前 scalar Zero/Nonnegative cones。SOC 若用于未来算法，必须按完整锥块处理；不能套用逐行常量删除规则。

## 验证与性能

初次接入时 44 项 native 测试通过：原 32 项行为案例、新增全固定案例、helper 11 项。原测试中对 solve 的调用同时走 baseline/reduced，比较全场与各报告后，再对 reduced 执行原行为断言。覆盖非零硬刚体、重叠柔性组、硬/软背景线、fixed points、共旋材料、局部梯度界、轮廓残差、不可行输入与零变量。后续 projection 功能的 46 项测试属于当前 solver 全量针对性测试扩展，非放松本功能容差。

预先保留的数值阈值：全场最大坐标差 ≤1e-4px、RMS ≤1e-5px，target/rigid/line 最大差 ≤1e-5px。当前三组同 binary、同输入配对均通过：

| 输入 | 变量数 | 约束行数 | baseline→reduced | 全场最大差 |
| --- | ---: | ---: | ---: | ---: |
| woman v2 +60 | 28,638→14,064 | 128,166→58,612 | 6.742s→1.485s | 2.853e-6px |
| user v2 +30 | 14,724→10,570 | 62,346→43,996 | 1.394s→0.618s | 3.033e-9px |
| dark v2 +30 | 20,758→14,648 | 88,326→60,768 | 2.501s→0.990s | 5.518e-7px |

这是一次本机配对，包含进程与 JSON 开销，不能当作稳定 P95、浏览器或 60 FPS 数据。三个性能输入尚未启用后续 projection，不将旧时长冒充所有新模型的时长。没有重新渲染照片；数值等价不替代各算法自身的图像验收。

## 重放与对账

`experiments/face-attachment-motion-20261110/build_reduced_field.py` 仅复用已有依赖编译当前场小 driver 和 native tests，不执行全项目 build。driver 支持 `--input ... --output ... [--reduce]`；`field-reduced` wrapper 明确启用 reduced 模式，供原 geometry CLI 使用。

`compare_current_reduction.py` 按当前 Problem 类型作成对比较，临时大体积场文件比较后清理。禁止直接用 2026-10-20 的旧 binary 解析当前问题，因为 Serde 可能忽略旧类型不认识的 fixed-points/material/contour 等字段。

对账证据：

- [性能审计](../qa/fixed-dof-performance-audit-20261113.md)；
- 本地 `test/results/face-usable-20261112/fixed-dof-independent-audit/report.json`：自由度/约束统计；
- 本地 `test/results/face-usable-20261112/current-reduction-parity/report.json`：配对结果与输入/输出哈希；
- 本地 `.run/face-field-current-reduced-20261113/provenance.json`：初次当前接入的源码、依赖和 binary 绑定。

照片、场数据、模型、缓存、二进制不提交。算法改动后应重新记录当前 binary 哈希，不把 wrapper 的哈希当作其调用 binary 的版本证明。
