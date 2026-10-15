# Gemini 建议核对与采用决策

日期：2026-10-14。依据用户提供的 Gemini 分析、当前源码、冻结版本回放和论文作者资料核对。建议是研究线索；下列候选没有通过实际输出验证前，不算已实现，更不能授予 8 分。最新“保直网格 + 局部自适应神经补全”的逐项数学评审、当前路线比较与露底补全实验见 [混合路线评审](hybrid-portrait-route-20261014.md)。用户已明确放弃 FLAME，本文关于它的核验仅保留为历史决策依据，不再开发。

## 当前失败与证据

用户原图的瘦脸正负方向均未通过：收缩有耳侧黑条、轮廓断接和皮肤碎片，扩张有阶梯拼接线。相关现行评分已撤销。两尺寸共 30 项冻结基线回放中，耳朵/饰品/头发固定像素及 RGBA alpha 改变均为 0；保护通过并不等于自然度通过。

1024 的 `faceWidth=60` 左侧 (361,335) 原 RGB `[149,112,94]` 变成 `[67,49,44]`；右侧 (648,384) `[181,134,114]` 变成 `[59,48,46]`。两点原面部/皮肤覆盖率为 1，耳朵/头发/饰物/脖子覆盖率为 0，距静止耳部约 7.6/6px。证据支持移动皮肤与静止连接处缺少衔接约束，不支持“同一像素保护与移动权限重叠”作为根因。完整记录位于 `test/results/user-face-contour-20261014/`，来源及可用性状态见 [用户案例](../qa/user-face-contour-20261014-review.md)。

调试采样 alpha 还显示，目标轮廓外出现不透明皮肤岛。近轮廓精确位移向 MLS/identity 混合后，一些外侧点重新采进原皮肤；补全层和这些皮肤岛交替，形成暗裂和锯齿。当前研究目标是使目标边缘、源边缘对应关系和覆盖率一致，同时保留耳根连接。此为待修复机制，尚不宣称所有裂缝只有一个原因。

不改计算的 trace 回放输出 SHA 与冻结基线完全相同：两侧代表暗裂原 alpha=1，采样覆盖变成 0/约 .002，进入空洞补全，再取暗 hair/scene 底层；不是 frozen 像素，没有背景估计 fallback，也在 jaw cap 的高度作用范围之前。两点均走近边缘 exact 分支。可复核消融、源码 patch 与图像见 [专项证据](../qa/gemini-geometry-ablation-20261014/report.md)。

## 可采用项

| 建议 | 核对结果与处理 |
|---|---|
| 区分语义类别置信度与物理 alpha | 正确且当前已区分。BiSeNet 覆盖率不是 matting；现有局部颜色线校正仍有失败，继续检验源/目标 coverage 的一致性。 |
| 检查耳根等移动/静止连接 | 与原图失败一致。保持原耳朵和饰品，约束邻接皮肤位移连续回到静止连接点；同时验证真实脸缘仍按方向移动。 |
| 检查 3px contour/MLS 混合与位移衰减 | 与调试中目标外皮肤岛吻合。优先隔离修正边缘传输，而非简单扩大平滑范围。 |
| 五官入口校验，避免缺区域时假成功 | 有价值；先追踪页面→graph→核心→输出。新检测的眼、眉、鼻、唇实际是 polygon，目前未证明 raster 格式导致用户失败。必须区分没作用、作用太弱、检测上下文陈旧。 |
| 妆容保留原亮度、唇纹和高光 | 是正确验收目标。现有唇色已做定亮度色度调整和亮中性反光衰减，不是毫无保护的 RGB 加色；仍未实现完整材质模型或证明自然度。 |
| 局部上下文裁剪与接缝融合 | 已有 server/client crop 选项、server Poisson。继续比较不同目标大小和上下文，不能把接口支持写成画质达标。 |
| 细褶与大褶皱分开验收 | 采用。现有衣物局部频率弱化不能代替结构去褶皱；保留未实现状态。 |

## 需要二次封装和实验的项

| 候选 | 所需工作与验收边界 |
|---|---|
| BBW / ARAP 网格形变 | 可作为 2D 连续形变候选。需固定边界/解剖控制、稀疏求解、三角形质量与翻转检测、Rust/WASM 性能和原图对照。BBW 与 ARAP 是不同算法，不能将现有 rigid MLS 更名为 ARAP。 |
| FLAME / DECA（已取消） | 用户已放弃，未接入、未下载权重，不再实施。历史核验：单图拟合不自动恢复隐藏纹理，身份 PCA 系数不是现成的独立“下颌宽度”滑块。 |
| 漫反射/镜面高光解耦 | 单张 RGB 的分解有歧义。先比较实用的高光保持与色度映射，再验证 BRDF/二色模型；没有法线、视角和光照证据时不凭空增加 Fresnel 高光。 |
| 黑色素/血红素分解 | 研究候选。普通 JPEG 的白平衡、压缩、妆容和光照会影响估计；不能把 RGB 色度投影叫真实色团浓度。验证红印减少、正常肤色、亮度与毛孔是否保持。 |
| 皮肤多频层与异常点修复 | 当前已有 `SkinDetailPlanes` 的 fine/medium/lighting、structure 和局部 blemish 路径，Gemini 所谓“只有两层”不准确。继续改善区分能力；频段不保证毛孔与瑕疵天然分离。 |
| Clean plate 人物分层重塑 | 当前已有人物/背景分离与暴露边带补偿。可升级为缓存底板，但优先补真正暴露的窄带；把整个人先删除会生成更大未知区域、增加错纹和时延。底板是推测内容，不能称恢复真实背景。 |
| 保直约束和密集人体/衣物先验 | 在宽袖、裙摆、椅背等遮挡图验证部位归属。原背景层优先保持原坐标；保直网格项不能授权弯曲或移动原本无需编辑的背景。 |
| Intrinsic decomposition 衣物去褶 | 反射率/光照分解有价值，但单图不唯一；衣褶还会改变轮廓和遮挡。条纹、纽扣、缝线不能保证 100% 保留，需已知 clean 参考及真实衣物对照。 |
| PowerPaint / 更新补全模型 | 作为模型替换候选，先固定 checkpoint、预处理、目标/protection、设备和延迟。模型名或论文年份不能代替与现有 LaMa 的同图比较。 |
| 高精度/宽色域 | 当前 8-bit RGBA 的确是限制；float 中间层、色彩管理、导入/导出需一起设计。先修可用性，不用引入 RAW/P3 回避当前裂缝。 |

## 不建议直接照搬的项

- **将 frozen 原像素保护改成软变形。** 这会移动或抹坏耳朵、首饰和用户冻结区。保留精确输出保护，让邻接皮肤形变连续衰减；自动类别误分另行诊断。
- **放开 skin/hair/neck/scene 任意混色。** 缝隙可能来自覆盖/层归属错误。盲目混色可能得到灰边、饰品拖影和脖子颜色污染；先查哪一层应当露出。
- **一律把 contour 过渡改成 6–8px 或全 exact。** 扩大带宽可能削弱形变或增加剪切。全 exact 消融在用户 1024 的 ±60 上仅改变 824/623 像素、最大通道差 3/1；两个主要 +60 暗裂样点完全不变。平滑 grid 切换也未改变主要暗裂；3px→8px 消融虽然改变更多边缘像素，两处主要暗裂样点依旧不变，实际裂口未解决。候选报告见 `test/results/user-face-contour-20261014/gemini-geometry-ablation/`。
- **将所有裁切阈值机械换成 smoothstep。** 平滑权重不修复错误 alpha、连接或供体类别；会使本不应裁切的皮肤被削弱。v11 jaw cap 只作用于下方 24%，不能解释更高耳侧的全部裂缝。
- **禁用 full context，强制每次 crop/Poisson。** 当前 street crop 曾产生更差的暗纹；上下文改变会改变生成结构。Poisson 匹配梯度和色差，不能恢复窗格/轨道，也不能还原丢失的原始高频。
- **承诺 BBW 全图 C1、无折叠，或 3D 拟合“天然解决所有遮挡”。** 线性三角形插值通常仅 C0，BBW 的连续数学目标不能替代离散场验证；BBW/ARAP 都需检查翻转。几何先验、图层遮挡和背景修复是独立问题。

## 文献信息纠正

已实际读取作者项目页/官方仓库。原建议的部分题名、作者及会议有误；这些多为经典方法，不能统称 2026 最新 SOTA。

| 来源 | 核对后的信息 |
|---|---|
| [FLAME 作者项目](https://flame.is.tue.mpg.de/) | 论文为 *Learning a model of facial shape and expression from 4D scans*，Li 等，SIGGRAPH Asia 2017。 |
| [DECA 作者项目](https://deca.is.tue.mpg.de/) | 论文为 *Learning an Animatable Detailed 3D Face Model from In-the-Wild Images*，Feng 等，SIGGRAPH 2021。 |
| [BBW 作者项目](https://igl.ethz.ch/projects/bbw/) | *Bounded Biharmonic Weights for Real-Time Deformation*，Jacobson 等，SIGGRAPH 2011；权重预计算后可实时混合，预计算求解不是免费。 |
| [ARAP 作者项目](https://igl.ethz.ch/projects/ARAP/) | *As-Rigid-As-Possible Surface Modeling*，Sorkine、Alexa，SGP 2007。 |
| [PSGAN 官方仓库](https://github.com/wtjiang98/PSGAN) | Jiang 等，*PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer*，CVPR 2020；是学习式妆容迁移，不是微表面 BRDF 解耦论文。 |
| [EleGANt 官方仓库](https://github.com/Chenyu-Yang-2000/EleGANt) | Yang 等，*EleGANt: Exquisite and Locally Editable GAN for Makeup Transfer*，ECCV 2022，非 CVPR 2022；同样不能当经典 BRDF 算法出处。 |
| [PowerPaint 官方仓库](https://github.com/open-mmlab/PowerPaint) | ECCV 2024，论文 *A Task is Worth One Word: Learning with Task Prompts for High-Quality Versatile Image Inpainting*，非 SIGGRAPH Asia 2024。 |

其余色团/固有图像/保直论文的泛化效果尚未在本工程重现，不因附件引用就记为实现证据。

## 本轮实施顺序

1. 修复用户照片瘦脸正负方向的轮廓外溢、耳根连接和补底；不调低强度掩盖失败。
2. 逐项复核五官/妆容真实 UI 操作与复合图层，补能让旧版失败的回归。
3. 同一输入、检测、参数、保护对比旧版与候选，原尺寸看脸缘/耳饰/背景及目标位移；再扩展到其他原始照片。
4. 接受有实际净收益的候选，更新 `doc/func` 和质量报告；未解决问题保持未通过。
5. 然后继续多频皮肤/妆容、身体补底、去人与衣物结构候选。维持无 AI 的手工编辑路径，学习式模型作为可选能力。
