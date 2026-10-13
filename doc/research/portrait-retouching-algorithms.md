# 修图算法：已验证的方法、复现与研究边界

日期：2026-10-13。以真实照片效果和运行效率选择方法；经典数学方法与现代学习模型结合。论文年份、开源代码和可用权重分别核对，不能把参考过论文写成完整复现。

## 已核对并采用

| 方法 | 一手论文/代码 | 当前实现及二次研究 | 仍需验证 |
|---|---|---|---|
| Rigid Moving Least Squares | Schaefer et al., Image Deformation Using Moving Least Squares, SIGGRAPH 2006；[论文](https://dl.acm.org/doi/10.1145/1141911.1141920) | Rust face_mls 实现刚性 MLS；以独立数学平移/旋转/控制点合同测试核对。脸部控制、固定五官、头发/饰物分层是本项目额外约束；边缘精算和稀疏位移网格降低计算量 | 实际脸型参数逐项自然度、倾斜/侧脸、组合参数及 WASM 热运行 |
| Fast Multi-Level Foreground Estimation | Germer et al., 2020；[PyMatting 官方](https://github.com/pymatting/pymatting) | face_matting 移植 F/B 局部正规方程和多尺度迭代，用于从混合边缘 RGB 分解前景与底层，避免把背景色拉进轮廓；不等于得到正确 alpha | 当前 alpha 来自解析覆盖，有误差；纸面方程测试只能证明公式，真实头发边缘需要照片核对 |
| Guided Filter / Fast Guided Filter | He et al. ECCV 2010、He & Sun 2015；[作者页面、论文及代码](https://people.csail.mit.edu/kaiming/eccv10/index.html) | Rust 皮肤改用有蒙版的 RGB/亮度局部线性模型；分析尺度有上限，原分辨率重新评价系数；尺度跟随脸宽，纹理由残差保留。取样隔离和蒙版准备是本项目封装 | 是传统保边平滑，不是祛痘、瑕疵检测或皮肤材质重建；需真实皮肤多图评价 |
| LaMa / Fourier Convolutions | Resolution-robust Large Mask Inpainting with Fourier Convolutions, WACV 2022；[论文](https://arxiv.org/abs/2109.07161)、[官方代码](https://github.com/advimman/lama) | 本地 CPU worker 使用已固定校验值的 IOPaint TorchScript 导出；989个参数和 BN buffer 与官方链接 checkpoint 全部相等。Rust 原生/WASM 提供统一目标蒙版与最终合成；Go/Web 按图层顺序插入模型推理 | 首张 full768去人好于 exemplar；512缩放和裁剪出现明显伪影，不能视作通用达到8分。模型效果、上下文和耗时分开记录 |
| Poisson Image Editing | Pérez et al., SIGGRAPH 2003；[论文](https://dl.acm.org/doi/10.1145/1201775.882269) | SciPy 稀疏线性求解在精确目标范围内匹配生成背景与原图边界，可选应用；保护区/外部/alpha严格保持 | 不能修复错误的生成内容；大目标的时间与内存需限制，默认不强制混合 |
| BiSeNet / CelebAMask-HQ | [BiSeNet论文](https://arxiv.org/abs/1808.00897)、[解析代码](https://github.com/zllrunning/face-parsing.PyTorch) | Web ONNX 分类分离脸部控制轮廓、整头、头发、脖子、皮肤、耳部、眼镜/耳饰。耳朵/饰物不进入脸型控制；置信度只作为编辑覆盖 | 分类覆盖不是真实 matting alpha；面部裁剪解析不是完整人体/衣物解析 |

## 继续评估，不直接依赖

| 方法/项目 | 用途 | 采用条件与当前结论 |
|---|---|---|
| [Structure-Aware Flow Generation for Human Body Reshaping](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Structure-Aware_Flow_Generation_for_Human_Body_Reshaping_CVPR_2022_paper.html) | 身形、腿、腰的结构感知流 | 已核开源与预训练线索；整图预测 flow 不保证固定背景，不直接把输出当本项目手工塑形结果。身体研究详见独立文档 |
| [ViTMatte](https://github.com/hustvl/ViTMatte) | 自然发丝和前景边缘 alpha | 官方为 Information Fusion 2024，有 trimap 与权重；需要可信 trimap、推理资源及真实边界 QA。暂未接入 |
| [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 精细人物分层 | 需要额外干净背景板，单张用户照片通常没有；不适合作为默认依赖 |
| [MAT](https://github.com/fenglinglwb/MAT)、[BrushNet](https://github.com/TencentARC/BrushNet)、[PowerPaint](https://github.com/open-mmlab/PowerPaint) | 更复杂的背景去人及语义补全 | 需要分别验证权重、环境和相同照片上的收益；CPU资源及推理时间限制不能忽略。未接入，不声称复现 |
| [ABPN / CRHD-3K](https://github.com/youngLBW/CRHD-3K)、[ModelScope ONNX适配](https://github.com/aoguai/skin-retouching-onnxruntime) | 学习的高分辨率皮肤精修 | 已核适配README的模型来源、生成器/局部瑕疵模型及导出需求。没有运行真实样片之前保持候选；不能用滤波器替代该模型的效果宣称 |
| [PPR10K](https://openaccess.thecvf.com/content/CVPR2021/html/Liang_PPR10K_A_Large-Scale_Portrait_Photo_Retouching_Dataset_With_Human-Region_Mask_CVPR_2021_paper.html) | 人像优先调色、组内色调一致性 | 官方目标是调色与组级一致性；不能把该数据集或论文当作瘦脸、去痘、褶皱重建算法 |
| Rolling Guidance Filter, ECCV2014 | 尺度分离、布料细纹 | [作者方法与原代码](https://www.cse.cuhk.edu.hk/leojia/projects/rollguidance/) 可用于进一步研究；细纹分离不代表重建严重折叠衣物，目前未复现完整迭代算法 |

## 效果驱动的研究流程

1. 明确失败照片及具体部位，保存原图、参数、蒙版、结果、差分和100%裁切。
2. 核对论文公式、官方实现与权重；说明直接使用、部分移植或本项目改进的范围。
3. 用已知几何/图层/纹理的独立测试验证方法，再用多张真实照片验证自然度。
4. 分别量检测、核心编辑、编码与冷/热运行；语义和固定背景的硬合同不为速度让步。
5. 新方法与旧方法使用相同输入比较；失败继续留在报告里。逐功能8分标准见[验收目标](../qa/quality-targets.md)。

传统液化、颜色、画笔、仿制与内容感知填补不依赖Agent、LLM或云服务。学习模型是独立可选推理路径；MCP/DeepAgent调用与用户手工调用使用同一操作协议，不另造一套图片算法。
