# FUNC-011：RBF/SVF 人脸轮廓与颈部连接实验对账

日期：2026-10-14。状态：**全部为隔离候选，未达到总体8/10，不合入生产/public WASM**。它们是本项目自己实现的可行轮廓与速度流管线，不是只罗列论文，也不代表复现某篇最新论文的完整系统。

算法：Wendland C2 compact RBF + regularized Cholesky、源轮廓控制五次 shooting、f32 RK4 正/反积分、同一 inverse flow 运输物理alpha；v16追加 Clarabel 凸QP约束源侧曲线，不添加新的侧反向/凹鼓包。RBF基础C2不等价于整个带raster gate的速度场C2。优先验证真实边缘位移、固定五官/耳/发/饰品和 scene，Jac/roundtrip只作为数值门槛。

| 版本 | 具体变化 | 结果与记录 |
|---|---|---|
| v14 | 用独立RBF速度替代 exponentiated legacy inverse displacement | 拟合更稳定，右下颊notch与扩张lobes，拒绝；[QA](../qa/face-rbf-v14/review.md) |
| v15 | 修正 selected-person class union 被误当成其他实例的5个自脸边缘固定像素 | notch因果改善，黑楔/扩张鼓包仍失败；2px无效果与5px阶段都保留；[QA](../qa/face-rbf-v15/review.md) |
| v16 | 可行侧轮廓QP，精确attachment和弯曲能量 | 扩张鼓包改善，六档颈部材质拼接失败；[QA](../qa/face-rbf-v16/review.md) |
| v17 | selection/protect共同约束，垂直chin/forehead跳过侧QP，情境固定与真实连接域分开 | 原绿人保护回归修复，原照六档与v16 byte exact，不算视觉提升；[QA](../qa/face-rbf-v17/review.md) |
| v18 | 从短连续可见neck边缘求source face连接，实际固定控制进入同一目标与速度 | 大黑楔缩小但残留材质硬接；+100实际20→8px，且独立单测37通过/3失败/1忽略，拒绝；[QA](../qa/face-rbf-v18-neck-attachment/review.md) |
| v19 | face+可信可见upper-neck联合seed，RGB refine物理alpha同流运输；下颈motion falloff，无硬neck pin | 黑楔/材质错接明显改善，+100 actual20.075px；四档仍残留皮肤岛/耳根点/硬边，未验收；[QA](../qa/face-joint-v19/review.md) |

原片始终是 `test/images/user-face-contour-20261014.jpg`，1024 canonical是同一原片；768不算新原片。没有重检测编辑后的图。各阶段保留源码、冻结native binary SHA、source/graph SHA、输出与局部失败图（v19仅四档），后续可准确对账。

黑三角probe只证明其颜色直接来自scene underlayer，不是隐藏neck材质真值。相应颈层延拓与LaMa负例：[材料诊断](../qa/face-amodal-neck-20261014/review.md)。最终可见轮廓须评估脸与静止颈部的联合结构，不能仅看face alpha或最大参数请求。

未完成项：多原照全部九参数视觉验收；同步共享目标、可滑动neck/ear连接与物理underlayer；WASM逐字节parity与实时效率。没有以通过的synthetic测试数量宣布产品级8分。
