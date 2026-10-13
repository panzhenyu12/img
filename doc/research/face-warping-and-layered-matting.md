# 人脸轮廓：MLS 形变与前景颜色分离

日期：2026-10-13。目标是移动真实皮肤脸缘，同时保持背景、原有头发、耳朵、耳饰和中央五官稳定。本文区分已移植算法、项目适配和未实现候选；算法出处不能代替真实照片效果验收。

## 已采用：Rigid Moving Least Squares

Schaefer、McPhail、Warren，**Image deformation using moving least squares**，SIGGRAPH 2006。[可复现参考代码](https://github.com/Jarvis73/Moving-Least-Squares)，[NumPy 公式实现](https://github.com/Jarvis73/Moving-Least-Squares/blob/master/img_utils.py)。本轮核对了参考实现的 rigid 模式；没有把尚未成功获取的原论文 PDF 写成已阅读。

`core/image-core/src/face_mls.rs` 实现二维 rigid MLS：使用逆平方距离权重（论文参数 α=1），计算输入/输出控制点的加权质心，再根据加权协方差归一化得到局部旋转。坐标始终保留浮点数。精确落在控制点上的点返回其目标，退化的零协方差采用平移，避免 NaN。

三个独立回归检查分别验证全局刚体变换、控制点/退化行为，以及与参考实现的论文矩阵表达式等价。参考 Python 中将采样坐标转成 int16 并裁剪图像边界的部分没有照搬，避免量化与越界导致的纹理跳变。

人脸工具将从面部类别 raster 的 .5 等值线提取的真实皮肤脸缘作为控制曲线，沿弧长均匀取 64 个控制点。眼、眉、鼻、唇区域添加固定控制点。反向映射使用“目标控制点→原控制点”，直接供逆采样，而非将前向场误当反向场。

以下属于本项目的二维人像适配，不是某个端到端人像论文的完整复现：

- 目标曲线按脸部滚转坐标系生成；9 个轮廓参数分别控制太阳穴、颧侧、下颌、额头或下巴。
- 3px 的轮廓边带采用对应曲线线段的精确逆采样，使 raster 边缘跟随控制曲线。内侧使用 rigid MLS。
- 位移随离轮廓的像素距离平滑衰减，五官固定区逐像素计算，不被网格插值穿透。
- 用 4px 懒加载采样网格缓存 MLS 场；距边缘约 8.7px 内仍精确计算，以保留真实边界。它是运行时优化，不是已实现 GPU 渲染。
- 原面部外像素在纯收缩时保持原样；混合外扩允许目标前景覆盖新位置。头发、耳朵、配饰类别是独立冻结层。

## 已移植：Fast Multi-Level Foreground Estimation

Germer、Uelwer、Conrad、Harmeling，**Fast Multi-Level Foreground Estimation**，arXiv:2006.14970，2020。[论文记录](https://arxiv.org/abs/2006.14970)，[PyMatting 官方算法源码](https://github.com/pymatting/pymatting/blob/master/pymatting/foreground/estimate_foreground_ml.py)。PyMatting 的 JOSS 2020 软件论文 DOI 10.21105/joss.02481 是另一份文献，不能混为该算法论文。

`core/image-core/src/face_matting.rs` 按官方多层实现移植：最近邻逐层缩放、small_size=32、小层迭代 10 次、大层迭代 2 次、regularization=1e-5、gradient_weight=1，并解每个像素前景/背景耦合的 2×2 线性系统。本实现采用顺序 Gauss–Seidel 更新，不依赖官方 Numba 并行运行环境。

输入必须是已经合成到背景上的不透明 RGB 照片，以及给定的 alpha。它估计 F/B 颜色，**不估计人物 alpha，不是人脸检测模型**。恒定前景/背景色的合成软边回归检查证明，该实现恢复的是前景颜色，而不是复制带背景颜色的混合边。真正带透明度的 RGBA 图片会返回 `None`，继续使用原 RGBA 分层路径。

实际面部前景颜色经 F/B 分离后再逆采样，可以减少重塑后的边缘携带旧背景颜色。v7改用不含头发/耳朵的`faceContourMask`作为可移动前景，完整head alpha仅提供上下文与保护。当前 alpha 由 BiSeNet 类别概率转换得到，它不是物理抠图 alpha；错误分割仍会限制 F/B 估计。对没有可见背景 donor 的少量内部不确定区域，使用估计背景颜色作为正则化补偿。它不能恢复被人物遮挡的真实纹理，不能称为生成式重建或干净背景复原。

脖子和场景背景仍使用独立的局部 donor 层：下颌收缩露出脖子时不能填成黄色墙面、白色背景。原外部像素不参加输出修改，局部传播和松弛仅给新露出的区域提供底层颜色。

## 2026-10-13 追加：分类覆盖率的局部颜色线校正

`face_matting::refine_alpha` 是本项目的有条件适配，不是将 PyMatting 的 F/B 估计改名为物理抠图，也没有声称复现 ViTMatte。它先寻找最多 12 个四邻接步以内的高置信前景/背景观测 donor（覆盖率 ≥.98 / ≤.02），在同类 3×3 邻域取颜色，再将当前 RGB 投影到该 F/B 颜色线：`a = clamp(dot(I-B,F-B)/dot(F-B,F-B),0,1)`。前后颜色差小于 25/255 的 RGB 向量，或拟合残差超过每通道 12/255 时保留原覆盖率；远处 donor、透明 RGBA、纯前景/背景也不强行改。

校正 alpha 供 F/B 估计和前景反采样使用；原覆盖率仍用于完整旧边带擦除与背景保护，避免 alpha 被校正为零后留下旧脸缘。合成“较宽的类别概率带、较窄的真实 RGB 混合边”回归验证其恢复真实 .25/.5 软边而不是把边缘直接硬阈值化。真实妆容样片中，外扩右脸的粗黄色边带明显减弱；细混色过渡仍存在。其他颜色、遮挡或低对比边界仍可能退回原 mask，不能据这一个成功样片宣称普适 8 分。

本轮还定位了与 matting 无关的旧下巴点线：face/neck 分类过渡同时低于 .5，被“其他人物”冻结规则误保留。已将明确其他人物证据限定到同时完全离开 face 两个支持域、且不含 neck 覆盖率的区域；在扩张前景真正覆盖、距原曲线 ≤3px 的未知原边带允许前景遮盖，但不突破用户/头发/耳朵/配饰冻结层。旧版本独立源码运行新增回归实际失败，记录见 [v3 QA](../qa/face-refinement-v3-review.md)。

新露颈部可以沿源/目标对应轮廓寻找已观察到的、完整 clean neck 双线性 donor。直接拷贝曾产生第二条硬弧线，已保留失败并改为按真正新露宽度逐渐退回重建底层；原可见颈部坐标和纹理不被几何拉伸。该补偿仍不是光照逆渲染或隐藏脖子的真实重建，强下巴缩短继续留未通过状态。

## 尚未采用的可复现候选

2026-10-14 v7–v9追加属于本项目适配，不是新增论文复现：

- 独立面部alpha参与F/B与反采样；可见头发作为固定底层donor，并将neck/hair/scene传播分为三层。原hair输出继续冻结，不能据此宣称完成发际线或前景长发遮挡重建。
- 脸部专用`refine_portrait_alpha`将高置信donor腐蚀2px，允许窄边带内饱和0/1类别覆盖率按原颜色线公式修正，保持内部确定alpha。新增合成测试使用真实三像素混合RGB配二值分类置信，旧`refine_alpha`确实无法校正，而新入口恢复软边；这不表示任意真实发丝都可被准确抠出。
- 五官固定区从扩张ellipse改为实际polygon内部与2px过渡，固定控制点落在实际polygon内。扩大ellipse的极值点原本会固定轮廓附近的可编辑脸颊，制造局部凹口。只修改冻结函数而保留旧handles的中间版本被新眼旁轮廓测试抓出（5px低于6px门槛）；最终同时固定实际polygon handles。

六图正负强值和原尺寸失败样片见 [v9 QA](../qa/face-polygon-v9/review.md)。这些改动改善层边界和局部连续性，不是三维解剖、学习式隐藏头发重建或普适8分证据。

| 算法 | 论文与实现 | 当前判断 |
|---|---|---|
| As-Rigid-As-Possible Surface Modeling | Sorkine、Alexa，SGP 2007；[作者项目页](https://igl.ethz.ch/projects/ARAP/)、[libigl](https://github.com/libigl/libigl) | 需要网格、局部旋转/全局稀疏求解。可作为更强的形变候选，当前没有移植，不能把 rigid MLS 叫 ARAP。 |
| ViTMatte: Boosting Image Matting with Pretrained Plain Vision Transformers | Yao、Wang、Yang、Wang；Information Fusion 103，2024；[官方代码/权重](https://github.com/hustvl/ViTMatte)、[arXiv:2305.15272](https://arxiv.org/abs/2305.15272) | 有可复现代码，依赖 image+trimap，适合探索改善细发丝 alpha。未集成；不能写成 ICCV 2023，也不能把阈值化分割叫 ViTMatte。 |
| Real-Time High-Resolution Background Matting | CVPR 2021；[BackgroundMattingV2 官方代码](https://github.com/PeterL1n/BackgroundMattingV2)、[arXiv:2012.07810](https://arxiv.org/abs/2012.07810) | 要求额外无人物背景照片。普通单张用户图片没有该输入，第一期不能直接依赖；论文 GPU 性能不能套用为浏览器 CPU 性能。 |

上述算法均提供可复现公开实现。Demo 阶段的选择仍按用户要求优先效果和开源代码；公开代码并不自动提供已训练的人脸形变模型，也不能保证任意遮挡、侧脸或大强度调整自然。

## 验收原则

自动化检查需覆盖真正脸缘移动、全部固定背景和发丝逐像素稳定、配饰保护、正负参数方向、五官锚定、RGBA、ROI、opacity 和滚转。真实照片需看原图/结果/差分，包含强下巴参数、长发、耳饰、眼镜和侧脸。

每项至少 8/10 是用户要求的完成门槛；本轮不以“来自论文”“测试通过”或“变化像素多”为 8 分证据。仍需逐参数真实图片效果验收，未达到自然度或缺少证据的项留在未通过状态。当前样片观察、失败和性能保存在 [脸缘目视核对](../qa/face-contour-visual-review.md)。
