# 同门槛、组件分母的新验收语义

此版本仅复用 50 holdout + 30 development 原始推理，不跑模型、不改 alpha、不调门槛。旧结果全部保留。改动是明确的任务语义纠正：提示框指向某个原 hair 连通组件，hair-core 保留率分母应为该组件的 `parser>=.9` 核心，而不是整张图全部 hair 核心。

正点>=.5、负点<=.1、可信skin保留<=5%、hair核心保留>=65%保持完全一致，仍在合格者里选官方score最高。组件没有可信核心则abstain。逐条验证源/graph/raw alpha hash、原组件面积与正点归属、重新计算旧全局保留率完全吻合。

| 数据 | 旧候选可选/总数 | 新组件候选可选/总数 | 变化 |
|---|---:|---:|---|
| 三张holdout | 32/50 | 50/50 | makeup小发束10组、dark右侧卷发8组从abstain变为可选 |
| 三张development | 28/30 | 28/30 | 选择完全不变，保留2个abstain |

这不是准确率或视觉通过率；旧mixed的整脸低alpha幽影仍可能通过同一二值门槛。仅纠正分母不保证物理matte。

makeup小发束814个可信core像素：负点combined-point候选2的旧全局保留率仅0.605%，实际该组件保留99.51%，skin mean alpha .00000363。新接受正确表达局部对象；全图union的hair保留率仍91.676%，因为大组件输出已经覆盖该小束，不夸大收益。

dark右侧卷发18,215个可信core像素：同协议候选2旧全局保留22.80%，实际组件保留88.14%。两组件逐个接受后，取原raw alpha最大值形成仅用于审计的union：

| dark负点协议 | 旧partial union全hair保留 | 新union全hair保留 | 新skin mean/p95 alpha |
|---|---:|---:|---|
| combined-point | 85.23% | 96.36% | .01007 / .0000406 |
| combined-box | 74.42% | 95.75% | .00913 / .0000914 |

原尺寸图证明右侧卷发大片缺失得以恢复。旧单组件失败保留，新的union仍有未覆盖的原parser核心和软边缘不确定性；parser不是真实alpha，不能把上述数字称为物理精度。部分灰白细边仍涉及原RGB混色，尚未做F/B颜色分解或实际warp；不声明卷发边缘已验收。

报告严格区分逐组件、global union和是否所有被提示组件已选。旧dark partial union即使整体hair保留>65%，还可能丢失另一组件的正点；新union另行运行原数值门槛。glasses有16个高置信hair像素属于未被本协议提示的组件，因此“所有被提示组件选中”也不等于所有源毛发已覆盖。所有union明确 `physicalMatteAccepted=false`。

所有80组、四候选指标、skin mean/p95/alpha>.05、parser fringe参考、旧/新拒绝理由在 `outputs/report.json`；变化组件的1:1图、旧/新union与原始f32 union另存。无生产改动、无新warp、无8分声明。
