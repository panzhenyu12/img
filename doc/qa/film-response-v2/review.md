# 胶片响应 v2：暖色色度肩部

2026-10-14。v1 独立实拍复核发现 Velvia 风格在微笑女性与暖光人像上明显偏橙。本轮只改变 Velvia 中等色度的红/黄色区域，其他五个新预设和 neutral 在六源照片上的输出逐字节不变。旧八个预设继续使用原算法。

连续 hue/chroma 权重限制额外饱和度及暖色 tint，保留绿、蓝和高色度景物的响应。这是颜色范围处理，**不是皮肤检测，也不是富士相机响应标定**；木材和相近暖色也会跟随变化。没有引入空间磨皮，原图纹理与 alpha 保持。

第一版较窄的色相肩部在 `190/183/125` 的单级绿色梯度上产生 3.174 RGB 级跳变，未通过连续梯度回归；[失败版本](../film-response-v2-initial/failure.md)保留。正式版本扩大肩部到黄色的过渡范围，原门槛不变，连续暖色梯度与景物色彩增强回归通过；六预设的严格灰阶单调回归也通过。

六张独立源照片 × 原片/六预设，共42项，最长边1024px、强度100、其他特效零。固定二进制 SHA `66eb1f7b1ca3578af4dfab9783d3149f225602cd311e93146e8b81a501bc17d1`，每次调用前后核对；完整 hash、原生首次/热运行、与 v1 的逐字节比较及裁切坐标见 [report.json](report.json)。100%裁切仅借用既有定位来定义展示范围；滤镜不运行解析模型。

原尺寸复核显示，微笑女性的额头/脸颊偏橙减弱，暖光女性的鼻唇周肤色更接近原片，深肤色明暗及毛发层次保留；青背景、绿色衣服、夜景灯光仍有饱和度增强。高强度 Velvia 仍有较强反差和暖色，不能作为所有人像的通用美颜色彩。Classic Negative 的负片色彩分离、Acros 的谱响应/银盐材质仍需继续改进。

| 代表样片 | 原片 / v1 / v2 原尺寸对照 |
|---|---|
| 微笑女性 | [100%](portrait-woman-velvia-100pct.png) |
| 暖光、头发遮挡女性 | [100%](portrait-hair-velvia-100pct.png) |
| 深肤色男性 | [100%](portrait-dark-velvia-100pct.png) |
| 带妆女性 | [100%](portrait-makeup-velvia-100pct.png) |

本轮视觉暂评 **7.7/10**：暖色自然度改善，风格/谱响应仍未完成独立目标对照，不记为完整8分。六图网格也保留：[微笑](portrait-woman-grid.jpg)、[暖光](portrait-hair-grid.jpg)、[深肤](portrait-dark-grid.jpg)、[带妆](portrait-makeup-grid.jpg)、[城市](cityscape-grid.jpg)、[黑白街景](street-crowd-grid.jpg)。原片已是黑白的街景不能验证彩色响应。

native 热中位单项104–448ms、全部非neutral中位186ms，现场有并行编译/回放；不能只取总体中位宣称每个preset都满足200ms。该时间含graph解析/校验/算子/合成，不含PNG、检测或WASM初始化。浏览器大图热运行和低性能设备仍未完成专项验收。

复现：`make quality-film` 使用当前核心生成回放；带固定v1输出的精确比较使用 `scripts/review_film_response.py --binary <冻结bench> --out <结果目录> --previous test/results/film-response-v1 --qa <文档目录>`。脚本需要Pillow/numpy。native/WASM、冻结和零强度的最终工程验证见 [current-validation](../current-validation.md)，合同测试不替代画质分数。
