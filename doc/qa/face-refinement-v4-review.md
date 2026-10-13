# 人脸下巴缩短 v4 实验记录

日期：2026-10-13。v4 在独立源码副本中验证下巴缩短的颈部照明连续性，尚未合入生产 `core/image-core/src/face.rs`。

## 变化

- 对已观察到、明确属于同一人物 `neck` 的区域做低频照明对应；原坐标高频纹理保留。
- 新露出的颈部沿目标轮廓到旧轮廓使用 smoothstep donor 混合，避免第二条硬下颌弧线；最后一版同时对原轮廓约 1.5–3px 的低置信混合边带做条件化去混色。
- 原 contour 近边带只处理 person/neck 高置信且 `faceContour` 为空的像素；正向 relight 增量限制为每通道 20，较亮的旧脸颈混合像素允许完整去除。
- `hair`、`ears`、`accessories`、用户保护、人物外背景和远处其他人物仍冻结。QA 脚本用 `--allow-neck-lighting` 显式放宽“语义 neck 可变”合同，并单独报告 neck 与场景变化。

## 证据

候选 executable：`.run/face-refinement-v4-sixth-image-render`，SHA256 `592fcedd3c9eb6b13f0cea3f6e524cbdd35820667a8420f0ea75af3194d4f1fc`。

报告：[24 项 chin 回放](face-refinement-v4-sixth/doc/replay-report.json)和[24 项 jaw 回放](face-refinement-v4-sixth-jaw/doc/replay-report.json)。六张 fresh 照片（`fixture-portrait`、`portrait-woman`、`portrait-makeup`、`portrait-roll`、`portrait-hair`、`portrait-glasses-street`）分别运行 `chin`/`jaw=-100,-60,+60,+100`。两组 24 项 ROI、头发、耳朵、配饰、保护和场景背景硬检查全部通过；scene background 修改为 0。语义 neck 修改计数独立记录，不当作背景保护通过。

32 项 face/MLS/matting 单测通过，1 项手工 benchmark ignored。合成回归确认旧下颌混色像素不作为静态 neck 纹理保留，并确认保护/场景不变。

## 视觉结果与限制

相比 v3，`chin=-100` 颈部轮廓附近的局部梯度下降：`portrait-makeup` 的 90/99 分位由 26.4/55.7 降至 20.9/29.5；`portrait-woman` 由 34.1/51.9 降至 27.6/31.8；`portrait-hair` 由 7.1/14.9 降至 7.1/9.8；`fixture-portrait` 由 71.0/150.5 降至 62.9/150.5。该指标只说明边缘连续性改善，不能代替多样本目视，也不能自动授予 8/10。

v4 仍需人工复核极强缩短、侧脸和遮挡颈部。合入生产前必须在父任务的 native/WASM 构建中重新生成同 SHA 逻辑的输出，不能复用本地实验 binary。
