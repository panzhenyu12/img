# 用户原图侧脸轮廓 v12 实验（未验收）

生产仍为 v11，v12 所有代码、二进制与调试输出保持隔离。完整最终候选为 `trimap-final-rejected.patch`；不是已实现或达到 8 分的功能。当前生产 `face.rs` 与实验开始前的 SHA 完全一致，未重建 public WASM。

## 实证与方案

用户 1024 原图源检测一次，六档 faceWidth `±30/±60/±100`，使用冻结 canonical PNG 与 graph；768 缩放派生图不是第二张独立原片。

1. 固定耳根与移动面颊目标存在连接冲突。仅加耳根几何约束减轻上部断连，不能消除下部条带。`attachment-only.patch` 为保留的失败。
2. `new-alpha` 在目标外出现不透明皮肤岛，来源是段对应到 MLS/identity 的衰减让目标外坐标重新采进原脸。目标外直接段传输减轻皮肤岛，但不能解决底板材质和耳饰避障。
3. 扩张目标包围固定耳环附近的背景保护像素会形成黑块。耳饰避障几何保持保护标准，仍允许无遮挡下颌扩张，但不能独立保证整体效果。
4. 耳根目标 source==target 后，原 `d=0..3` 的 exact/MLS 混合仍能在物理 RGB fringe 里产生非零位移。保留物理 2px trimap 的 exact 对应后根部细缝明显改善。这个结论只在已有固定目标约束的候选成立，不等于 baseline 的 8px 混合带消融有效。
5. 干净侧脸支持下几何 alpha 对齐可以平滑轮廓，但较早候选在根部生成点状黑线；必须保留拒绝记录。
6. 颈部外轮廓延伸只能为隐藏材质提供先验。当前硬猜测和局部一像素抗锯齿仍有下颌/颈部拼接与平面阴影，**拒绝作为生产算法**。

## 当前结果

六档原图 fixed 耳/耳饰/头发、实际五官内部、纯收缩 scene 背景变更均为 0。4 张独立其他原片（woman/makeup/hair/roll）双向 ±100 的 8 case 硬保护均通过，见 other-portrait-replay/replay-report.json。31 个 face 单测通过，1 个手工 benchmark ignored；两项新回归在冻结旧代码中失败。保护及单测通过不提供主观评分。

原图中上部耳根连接、成片侧脸皮肤岛及扩张耳饰黑矩形改善；下颌与颈部/场景衔接、细暗边及照明仍不符合可用标准。没有给出 8 分评价，不接生产。

## Gemini 对照

[Gemini 消融报告](../gemini-geometry-ablation-20261014/report.md) 证实全 exact grid 或平滑 grid 无法修复原始大裂缝，8px exact/MLS 带也不能解决原始耳根连接。原黑点分别来自 hair/scene 底板，不是 fallback；v11 下巴 opacity 门限未执行到耳根。禁止通过软化耳饰冻结或跨材质大范围 RGB 平均处理。

## 最小必要后续改造

几何采用单一 source-forward 连续场，目标与 inverse sampling 由同一个场生成，selection/protection/附件约束进入场，不能在最终位移上另乘硬权重。优先隔离验证 SVF `exp(v)` / `exp(-v)` + f32 RK4，测局部 Jacobian、双向误差、轮廓实际幅度、保护域与材料层。

保留 face/neck/hair/scene 分层，把露底带按目标主体 coverage 与露出的材质约束分别重建；自动神经补全必须禁止把旧脸填回露底带。当前硬颈先验不作为后续方案基础。

## 文件

`source-provenance.json` 保存源和冻结二进制 SHA；`user-replay-report.json` 保存六档 RGBA 合约和输出 SHA。最终完整代码差异以及逐步失败差异均保留为 patch；真实效果失败截图保留在本目录，原始调试 alpha/geometry/全图在 `test/results/user-face-contour-20261014/1024/*-candidate`。
