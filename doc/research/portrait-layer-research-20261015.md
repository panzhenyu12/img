# 单图人像分层研究与实测（2026-10-15）

**决定：当前继续以 ZIM 为已验证的候选 alpha 源；SAM2Matting Tiny 已真实跑通 CPU，但三个样片都出现确定皮肤区域透明化，拒绝全域替换。** 下一步优先验证当前 source ownership + delta compositor，再以 MEMatte 做不同模型家族的 trimap 对照。仅换一个更新的抠图模型不能推导出耳饰遮挡、真实隐藏背景或自然瘦脸。

本次为有边界的候选筛选与可行性实验，非穷尽系统综述。使用 literature-review 与 scientific-critical-thinking 技能记录检索、输入/输出、反证与限制；未调用付费 API，未外发原人像。除 SAM2Matting 外，新候选没有本机推理结果。

## 检索与复现范围

日期：2026-10-15，Asia/Shanghai。范围为 2024–2026 单图 matting、source layer decomposition 和局部人体编辑的遮挡边界。纳入：官方公开推理代码、权重链接、单图输入，能只提供 alpha/层而保留原始 RGB 的路径。排除：额外拍摄、只有数据集、未公开推理、整图生成人像替换、视频专用且无图像入口。

- CVF：读取 CVPR2024、CVPR2025、ICCV2025 的全部题目页，按 `matting`、`layer decomposition`、`human instance` 筛选，再访问相关论文记录。
- arXiv API：`(ti:matting OR ti:"layer decomposition") AND submittedDate:[202401010000 TO 202610152359]`，按提交日期降序、最多 25 项。查询有材料科学等同词噪声，人工筛去；该上限也会漏掉较早论文，不能声称领域覆盖完整。
- GitHub repository API：`matting CVPR 2025`、`matting ECCV 2024`、`matting 2025 in:name,description`、`image matting 2026`、`Matting Anything`、`layer decomposition image 2025`、`human reshaping 2025`，再以 `SDMatte`、`RenderMatte`、`MattePro`、`FCLM matting`、`LayerDecomp` 等名称追溯官方仓库。
- 官方 README 指向的 Hugging Face API 仅核对公开状态、revision、文件大小、LFS SHA；下载了唯一一个 SAM2Matting Tiny 权重。

[检索与原始证据目录](sources/portrait-layer-20261015/)保存查询 URL、返回记录、CVF 摘要、arXiv 正文与固定代码快照。CVF 全量索引约 23MB，只保留响应 SHA、大小与筛出的 primary 页面，避免把无关全文索引放进项目。原生代码与权重检索不是独立模型验证。

## 四个可执行候选

| 候选 | 已核对的原理与可用性 | 对本项目的价值与限制 | 本轮结论 |
|---|---|---|---|
| **ZIM / ICCV 2025** | [CVF论文](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_ZIM_Zero-Shot_Image_Matting_for_Anything_ICCV_2025_paper.html)；prompt-aware attention + hierarchical pixel decoder；[固定源码](https://github.com/naver-ai/ZIM/tree/392e613c8a3d395ce4dda3b51edb6dabc1fd0004)，本项目已下载官方 ONNX 权重并实跑 | 预测指定目标 alpha。**不预测 skin/metal/ear 的遮挡顺序，也不恢复隐藏 RGB。** full/crop 与多提示提供 disagreement 证据；不能当真值投票 | 已有基线，继续审 ownership/合成合同 |
| **SAM2Matting / 2026** | [arXiv正文](https://arxiv.org/html/2606.27339v1) §3.2–3.4：学习 ROI、pseudo-trimap、progressive alpha；[固定图像入口](https://github.com/FudanCVL/SAM2Matting/blob/73dd721d77b56749248aefe5e8824d7f61b9d13c/inference_image_sam2.py)；[公开Tiny权重](https://huggingface.co/FudanCVL/SAM2Matting/tree/4315db9c60d27fde396b09765748a0ca6c97bed5) 215,569,778B，未 gated | 官方支持单图 mask，且确实能在本机 CPU 跑；学习 ROI 能改变输入 mask 的确定区域，当前把额头/颈底误当不确定域。官方 `fgr.png` 只是 `原RGB×alpha + 绿底×(1-alpha)`，**不是去混色后的 foreground RGB** | 本轮已跑四次；raw 全域拒绝；可保留为独立对照 |
| **MEMatte / AAAI 2025** | [arXiv:2412.10702](https://arxiv.org/abs/2412.10702)；adaptive token routing + lightweight token refinement；[固定代码与公开checkpoint链接](https://github.com/linyiheng123/MEMatte/tree/3c887e2a517b38f936b97f27d933c5897c6d47a1) | 不同于 SAM 系列的 trimap 对照，可将有争议皮肤/金属边置 unknown。需要 PyTorch/Detectron2，已知前景/背景被强制 clamp；错误 hard trimap 无法靠模型纠正。Drive 文件本轮未下载验 SHA | 下一项优先做局部 crop 对照，尚未实跑 |
| **SDMatte / ICCV 2025** | [CVF论文](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_SDMatte_Grafting_Diffusion_Models_for_Interactive_Matting_ICCV_2025_paper.html)、[正文§3](https://arxiv.org/html/2508.00443v1)明确 **one-step deterministic alpha prediction**；[固定代码](https://github.com/vivoCameraResearch/SDMatte/tree/c3ea6949f270d253539a2bda2ad290e127934de9)、[Lite权重](https://huggingface.co/LongfeiHuang/LiteSDMatte/tree/7f527b9a69b84cc6dce311e5cb88b2687c19af33)公开未 gated | 利用 diffusion U-Net 表征并接点/框/mask，**不是 LaMa式 RGB补洞或逐步整图重绘**。因此不能因“diffusion”一词套用本项目 LaMa 生脸失败。Lite checkpoint 实际 8,459,343,160B；Detectron2/GPU 路径较重 | 可做独立服务端对照；当前不为它下载 8.46GB 模型 |

论文称一般 matting 更好，不等于“皮肤子部件 + 耳饰”可被正确识别。这四者的保身份能力来自只替换 alpha、继续使用源 RGB/确定性合成的工程合同；不是模型给出的身份 100% 保证。

## SAM2Matting Tiny 实际结果

代码固定 `73dd721d77b56749248aefe5e8824d7f61b9d13c`，权重 SHA256 `5b9321e3b51bc20f5b84c208746cc083dd3053dd701590f2e88dc8640afcc39d`，保存于 `.models/sam2matting-20261015/`。官方 requirements 的 TensorRT/torch2.8 没有直接安装：复用当前 torch2.2.2，在实验私有 `deps/` 加 torchvision0.17.2、Hydra、iopath 等；原 vendor 不变，CPU float32，3 线程，禁 autocast/compile。checkpoint missing/unexpected keys 都为 0。这是本机兼容性实验，不表示所有后端受官方支持。

首轮使用旧 `outputs/full-point` idx2，用户图额头出现透明块。为排除 seed 差异，又统一用 **source-only `prompts-v2/<name>/crop-point` idx2 > .5** 二值 mask，full source RGB，一图一次 encode/decode；没有新检测、人工结果裁剪或 semantic 后过滤。

| 样片 | encode / decode | full+crop 同时≥.99像素中被降到<.9 | 被降到<.5 | 1×观察 |
|---|---:|---:|---:|---|
| 用户原图1024² | 3.95s / 5.31s | 16,844 / 137,808 | 9,853 | 额头和颈底明显透明，脸缘变软/局部锯齿 |
| woman 1280×1920 | 3.60s / 3.96s | 7,273 / 241,630 | 375 | 额顶变薄，轮廓更柔；未证明优于原alpha |
| hair 588×768 | 3.66s / 4.05s | 1,214 / 42,474 | 560 | 额头透明块、颈底软边 |

三图批次 peak RSS 2.92GB；这是当前机器的单次实验，不是浏览器性能或统计分布。full/crop 同时高 alpha 只是支持参照，不是 ground truth；额头明显透明还经过原尺寸图确认。`full-point` 与 `crop-point` 二值 seed 本身分别有 16,598 / 5,219 / 2,383 像素不同，不能把不同 seed 的结果当纯模型 A/B。

[本机原始结果](../../experiments/sam2matting-tiny-probe-20261015/outputs-v2-crop-seed/report.json)、每图 `alpha-raw.npy` 与 `face-1x.png` 均保留，**accepted=false**。Alpha preview 不等于瘦脸结果，未宣称消除 ghost；要判断新 alpha 在形变后是否更好，必须另做固定 V37 geometry + compositor replay。

## 未进入立即候选的具体理由

- **RenderMatte，arXiv:2608.08487**：[代码](https://github.com/Nislab-r/RenderMatte/tree/b7ed2505e8819fee4b4c167cd95093cdf9db241f)和 SFT/LoRA 公开，但 SFT 23.8GB，基础 FLUX.1-Kontext-dev 的官方 API 标记 `gated:auto`；不是当前无认证、轻量可复现链。要求 trimap，硬 clamp 仍会锁住错误。保留研究，不把“exact-alpha训练数据”当推理真值保证。
- **LayerDecomp / CVPR2025**：[论文](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Generative_Image_Layer_Decomposition_with_Visual_Effects_CVPR_2025_paper.html)确实研究 clean background + 带视觉效果的 transparent foreground；[官方仓库固定fb92fc2](https://github.com/Rayjryang/LayerDecomp/tree/fb92fc2ac8dd5060db79fb9cdffcb41099719535)仅有短README和图片/项目页材料，未提供可运行推理/权重。不能接入。
- **MP-Mat / ICLR2025**、**FCLM / AAAI2026**：官方仓库仍声明整理/公司审核，tree 未见推理。不是可立即复现的模型。
- **MaSS13K / CVPR2025**：[primary摘要](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_MaSS13K_A_Matting-level_Semantic_Segmentation_Benchmark_CVPR_2025_paper.html)仅七大类 human/vegetation/ground/sky/water/building/others，不能直接区分脸皮/耳饰。是可用的高分辨率语义工具，非本问题的细部归属解法。
- **MattePro**：代码、Drive checkpoint、交互 SAM2+MEMatte 组合可用，但 README 的论文入口仍为 `arxiv-incoming`；不是额外一篇已核对论文。可作为 MEMatte adapter 参考。
- **Parallax Portrait Matting**明确需要第二帧；SSMatte/RefLayer/PixelART 本次未获得与问题匹配、完整可复现的官方链。前者输入不符，后者记录为未验证，未对方法效果下结论。

## 可执行下一步与拒绝标准

1. **当前主线**：固定 source/graph/inverse，先比较 ZIM direct raw + delta compositor 与旧重建。0 参数逐字节 identity、未动区域逐字节保留、skin移走与destination重建必须同时检查。RGB重投影误差小不证明 alpha/层归属唯一。
2. **模型对照**：SAM2Matting 已有 raw 只供离线 replay；不要把额头失败用不记录的硬裁剪遮掉。下一模型用 MEMatte：确定皮肤/金属/背景证据制作 trimap，争议区置 unknown，沿同一几何重放 ±100。只与同 seed/同crop/同合成器比较。
3. **缺失层的处理**：skin、hair、ear、metal、scene 分别保留 observed source support、unknown 和 destination write 域；低 accessory alpha 不自动成为 skin，低 skin alpha 也不自动成为 scene。先用双模型皮肤缺席 + 新金属alpha缺席 + RGB连通 scene 支持，隐藏背景保留“估计”标记。
4. **验收**：用户、woman、hair 至少三图正负强值看原尺寸轮廓、耳饰、发丝；记录内部不透明区域丢失、保护区覆盖、source清除残条、destination断裂与性能。mask benchmark/模型年份/测试进程成功均不能换算成 8 分。

方法引用：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://arxiv.org/abs/2609.00065)，[DOI](https://doi.org/10.48550/arXiv.2609.00065)。本日 API 确认为 v2，未列期刊 DOI；这里只致谢本报告的检索/批判流程，不将其当成图像算法证据。
