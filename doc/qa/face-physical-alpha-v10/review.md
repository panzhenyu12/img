# Face physical alpha v10 候选评审

当前状态：隔离候选，生产源码尚未修改。第二候选修复“二值解析外缘保留旧物理轮廓”的可重现问题；已通过全部既有保护回放。它没有消除 makeup 强档耳下的固定底层接缝，不能据此给脸型功能评为 8 分。

## 问题与修复范围

v9 的 portrait trimap 已能恢复解析 alpha=0/1 边缘的真实混合 alpha，但原始覆盖门限、holes 和 reconstruction 供体仍使用解析置信度。这会冻结实际包含皮肤的外缘，并把混合像素作为清洁背景供体。候选让这三处一致采用 refined physical alpha，并在修正后确定固定 neck/hair 底层。

第一候选的全局颜色投影会误改 fixture 中 5 个原 head/neck 外部点（其中 4 个明确 scene 点），并破坏 head-only 旧路径的邻近人物保护，因此已拒绝。完整 72 项失败报告、补丁、测试失败日志及图像保存在 attempt1-rejected，未减少任何回放保护条件。

第二候选保留原 head 上下文约束：head 与 facial class matte 不同时，head alpha=0 的外部支持禁止颜色投影增加前景；二者为同一 face tracing 时，允许恢复实际混合外缘。没有 facial contour matte 的旧 head-only 路径保留原 .5 输出门限。五官、头发、耳饰、其他人物和用户冻结逻辑保持原有判断。

## 已知真值回归

新增 Rust 测试 saturated_binary_parsing_removes_old_physical_fringe_with_correct_background。输入由已知 F=[220,155,100]、B=[20,40,60] 和 3 px 物理 alpha 混合生成，但解析置信度被二值化。它验证真实内轮廓至少向内移动 4 px、旧混合边及完全暴露带的每通道背景误差 ≤3，并逐像素验证所有 true alpha=0 背景完全不动；它不是仅检查“输出发生变化”的测试。

在中心行，v9 原外缘 x40/41 保留 [46,55,65]/[113,93,79]，暴露带 x42/43 错填约 [110,92,78]。候选将这些点准确恢复到 B=[20,40,60]。候选的所有 pure true-background changes=0。全图 628 个旧外缘混合点中，v9 全部原样保留；候选仍有 92 个未改，主要是此次 faceWidth 未移动的额头端，不能要求未移动端也消失。比较图与逐像素真值记录见 mixed-edge-comparison.png、mixed-edge-metrics.json。

## 真实照片与保护

使用既有 6 张照片、faceWidth/jaw/chin、±60/±100 共 72 项；保护检查通过 72/72。固定 binary 的 SHA-256 校验在整个回放中保持一致，72 项仍要求 ROI、五官 polygon 内部、hair/ears/accessories、用户保护以及收缩时 scene/head-neck 外部满足既有条件；允许的既有 neck lighting 合同没有扩大。

专项 Rust face tests：29 passed、1 ignored（手工性能 benchmark）。补丁 git apply --check 已通过。

已目视 makeup/woman/hair 的 faceWidth 60/100 原尺寸三联图。第二候选没有看到新增大块伪影，但对真实照片的变化整体较小。makeup 耳下 layer2 hair 与 layer0 scene 的台阶仍明显：faceWidth=60，(477,300) 从 [160,108,77] 变成 [163,109,79]，下一行仍为 [211,188,78]；这种约 79 的绿色通道跳变仍失败。makeup-layer-join.json 与 ear-join-4x 图片保留了这个问题。hair 的强档耳下/颈部背景色楔也仍存在，不宣称已解决。

CLI 端到端中位数：v9 533.01 ms，候选 486.27 ms；包括图像/JSON 解码、校验、kernel、PNG 输出，并发机器工作影响时延。最大候选 1145.12 ms。该数据没有证明稳定的实时浏览器性能。

## 冻结输入与交付

- baseline: .run/face-polygon-v9-final-image-render
- baseline SHA-256: 58013da0cc9840dd8d4525e9d49ee7a150ba1a02b7f225c26ad525aa21cbbeb1
- candidate: .run/face-physical-alpha-v10-attempt2-image-render
- candidate SHA-256: 3b36aaaa0f3b75521cc8f7726a4a3b605619b2e95c9b69818c3cac9af9d718f8
- isolated source: .run/face-physical-alpha-v10-core/src/face.rs
- isolated target: .run/final-core-review-target
- apply patch: doc/qa/face-physical-alpha-v10/face-physical-alpha-v10.patch
- raw outputs: test/results/face-physical-alpha-v10-attempt2

建议将它作为已知 alpha 一致性缺陷的窄范围修复评审。真实复杂底层重建与耳下接缝仍需继续研究，不能将本轮通过的保护测试当作完整画质达标。
