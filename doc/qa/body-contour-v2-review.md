# 身体轮廓 V2 实拍逐项复核

日期：2026-10-13。五张实拍、15 项单部位、参数 60，最长边 768px。**本轮 15 项均未达到逐功能 8 分目标，不能宣称身形效果已经完成。**

已逐张查看原图、输出、差分与局部放大；评分是这批照片上的主观工程评审，不能外推为某项功能在所有照片上的质量。模型检测使用原 Pose 缓存，编辑使用当前 native CLI；支持域由当前代码重新测量。

15 项都可渲染，支持域外变化均为 0；原已知背景变化的失败项 0。CLI 总耗时 640–1188ms，包括加载与 PNG 输出；不包括本次未重跑的检测，没有 1024px、初始化或峰值内存证明，因此性能分只是本组编辑的初步评价。

硬条件：真实轮廓改变、固定原背景、非目标部位不误动。评分维度依次为效果/自然度/部位保护/编辑效率（权重 35/35/20/10）；每维至少 7 且总分至少 8 才通过。**背景差分零不能掩盖手、发、包、袖子或持物误动。**

| 实拍 / 部位 | 四维评分 | 总分 | 原背景变化 | 目视结果 |
|---|---|---|---|---|
| [fashion-standing / waist](body-contour-v2/fashion-standing-waist.png) | 8/6/7/7 | 7.00 | 0 | 真实衬衫外边收缩，中轴和手基本保持；衣袖贴腰处部位边界仍不精确，轮廓窄带有细色边。 |
| [fashion-standing / left-upper-arm](body-contour-v2/fashion-standing-left-upper-arm.png) | 8/6/8/7 | 7.20 | 0 | 袖子外边确实内移，布纹保留；低分辨率软边与填补窄带有浅色重边，缺少真袖子/躯干像素分离。 |
| [fashion-standing / left-thigh](body-contour-v2/fashion-standing-left-thigh.png) | 8/5/8/7 | 6.85 | 0 | 裤腿真实收窄，手与骨轴保持；腿间背景窄带有残影，植物/格栅结构的延拓不可靠。 |
| [fashion-standing / right-upper-arm](body-contour-v2/fashion-standing-right-upper-arm.png) | 8/6/8/6 | 7.10 | 0 | 弯曲上臂外轮廓改变，袖口和前臂总体保持；轮廓转折和背景延拓处有细边，仍不够自然。 |
| [fashion-standing / right-thigh](body-contour-v2/fashion-standing-right-thigh.png) | 8/5/8/7 | 6.85 | 0 | 裤腿外侧真实内移；腿间窄带与地面结构线有细条和接缝，不能以背景原像素不动证明自然。 |
| [fullbody-skirt / waist](body-contour-v2/fullbody-skirt-waist.png) | 4/7/7/6 | 5.85 | 0 | 腰侧被衣袖连接/遮挡，主要改变裙子内部纹理，可见真实腰外边变化不足；不能把支持区内差分当成收腰成功。 |
| [fullbody-skirt / left-upper-arm](body-contour-v2/fullbody-skirt-left-upper-arm.png) | 7/6/8/7 | 6.85 | 0 | 可见衣袖外边略收窄，蕾丝细节大致保留；外边有暗边/小折线，隐藏内侧只能固定。 |
| [fullbody-skirt / right-upper-arm](body-contour-v2/fullbody-skirt-right-upper-arm.png) | 8/6/8/7 | 7.20 | 0 | 衣袖真实外侧收窄；浅色袖边与背景有薄色缝，长袖下面的手臂轮廓并未被重建。 |
| [fullbody-side-fashion / waist](body-contour-v2/fullbody-side-fashion-waist.png) | 5/3/3/7 | 4.10 | 0 | 垂挂大衣及包侧区域一起被缩，目标腰部隔离失败；外套软边残影显著，缺失区有拖影。 |
| [fullbody-side-fashion / right-upper-arm](body-contour-v2/fullbody-side-fashion-right-upper-arm.png) | 6/3/3/7 | 4.45 | 0 | 上臂外边改变，但连在主体内的手/头发及大衣被误动，靠脸手部出现重影；必须判失败。 |
| [tilted-shirt / waist](body-contour-v2/tilted-shirt-waist.png) | 6/5/4/7 | 5.35 | 0 | 斜姿和宽袖的轮廓收缩明显；同时改变袖子/外套而非隔离真实腰，软边与窄带有残影。 |
| [tilted-shirt / left-upper-arm](body-contour-v2/tilted-shirt-left-upper-arm.png) | 8/6/8/7 | 7.20 | 0 | 宽袖外轮廓真实内移，衣料纹理基本连续；袖边略有细重边，仍缺精确部位 alpha。 |
| [action-suit / waist](body-contour-v2/action-suit-waist.png) | 7/5/4/6 | 5.60 | 0 | 外套腰侧真实收缩，但电脑上缘附近出现差分和接缝；持物保护没有可靠语义隔离。 |
| [action-suit / left-upper-arm](body-contour-v2/action-suit-left-upper-arm.png) | 8/6/7/7 | 7.00 | 0 | 抬臂上下轮廓局部内移；肩接缝和弯曲袖边有折线，外侧背景填补色调略不连续。 |
| [action-suit / right-upper-arm](body-contour-v2/action-suit-right-upper-arm.png) | 8/6/8/7 | 7.20 | 0 | 上臂外轮廓真实改变，前臂保持；袖边和腋下仍有窄带接缝，不再因可渲染判通过。 |

## 整改所得与未解决问题

相比旧仅改背景/衣物内部的算法，宽衣腰 support 已跨越真正外边、肢体两侧独立测量，轮廓确实能内移；完整软 fringe 跟随映射，原已知背景保持；实际目标部位 alpha 接口已可用。缺 mask 的整片 warp 路径已删除。

剩余失败集中在三类：主体 confidence/软边不是准确 alpha；宽袖/躯干/外套相连导致缺少真 partMask；收缩窄带的背景延拓无法可靠恢复植物、格栅、建筑线和腿间结构。侧身图的手/头发重影、宽衣腰和电脑误动属于部位隔离失败，必须优先修。

当前自动检测尚不生成真 partMask。FASHN 实际模型/ONNX 仅实验未接产品，且同类上衣也分不开宽袖与躯干，不能把接入模型当作验收完成。需要可校正的真部位 alpha、手/头发/饰物/持物保护及更精确边缘；复杂窄带补全单独验证。

裙下/交叠腿和不可见小腿仍不开放自动独立重塑；肩宽、腹部、胯部、前臂、小腿、全身组合仅有合成性质测试，本组没有逐功能实拍评分，不得从这 15 项推断全部达标。身高、腿长、体态与衣物大褶皱仍需独立算法。

完整原图/输出/graph 在 `test/results/body-contour-v2/`，局部证据在同名 QA 目录。版本摘要见 [JSON](body-contour-v2-review.json)，自动差分见 [metrics](body-contour-v2-metrics.json)。用 `node scripts/replay_body_quality.mjs` 与 `python scripts/review_body_quality.py` 重生成；后者需要 Pillow，评分常量必须在目视复核后更新，不能由程序自己提高分数。
