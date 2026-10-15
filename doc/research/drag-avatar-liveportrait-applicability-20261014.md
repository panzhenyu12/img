# Drag、3D Gaussian 与 LivePortrait 对修图产品的适用性核验

核验日期：2026-10-14。前六节为建议中六个项目的官方论文、仓库说明和关键源码只读研究，未运行这些模型。第七、八节已追加较新候选，并有独立 Inpaint4Drag/ZIM 实验；详见 [2025–2026候选研究](portrait-candidates-2025-2026-20261014.md) 和 [Inpaint4Drag负例](../qa/inpaint4drag-20261014/report.md)。下文作者性能数字均不能代替本项目浏览器、当前 Intel Mac 或 CPU 服务的实测。

结论：建议中的“可控二维形变 + 局部神经补全”值得作为修图架构方向，但列举的生成/头像项目不能直接替代脸型、五官、身形的参数编辑核心。Drag 系列适合另设生成式拖动工具；LivePortrait 可用于表情和姿态；FlashAvatar、LGM 面向头像或三维内容构建，与单张照片的精确美颜编辑目标有明显距离。FLAME 已按用户要求放弃；FlashAvatar 本身仍依赖 FLAME，因此也不应作为当前基础路线。

这些代表论文发布于 2023–2024 年，LivePortrait 核验到的论文修订为 2025 年。它们的公开代码值得研究，但不能据此称为“2026 年最新技术”或断言已有最高效果。本次没有完成对 2025–2026 全部相关论文的横向检索。

## 一、真实输入和任务

| 项目 | 官方输入和处理过程 | 与本产品功能的关系 | 当前建议 |
| --- | --- | --- | --- |
| DragGAN，SIGGRAPH 2023 | 原生编辑 StyleGAN 生成图；真实照片要先通过 PTI 等 GAN inversion 得到 latent 和适配权重 | 用户给点对可移动鼻子、下颌等语义位置；没有现成“瘦脸/瘦腰”参数体系。真图重建可能先损失身份、纹理或场景 | 借鉴点对控制；不直接用作全部美颜的实时底座 |
| DragDiffusion，CVPR 2024 | 真实单图先训练该图 LoRA，再 DDIM inversion、latent 梯度优化与参考控制去噪；生成图可复用生成 latent | 通用图像内容拖动；不是逐项美颜模型。论文的 identity-preserving 主要指输入图像内容保持，不能自动解释为人脸识别身份达到产品要求 | 可作为可选 GPU 生成式拖动能力 |
| FreeDrag，CVPR 2024 | StyleGAN 分支也需要真实照片 inversion；Diffusion 分支对每张真实照片训练 LoRA | 自适应 feature template、回溯线搜索改善拖动丢点与误跟踪；仍存在二维拖动语义歧义 | 借鉴稳定跟踪和回溯；与 DragDiffusion 一起作可选实验 |
| FlashAvatar，CVPR 2024 | 短单目视频、分割/alpha 和跟踪结果；对每个身份训练头像；Gaussian 嵌在动态 FLAME 表面 | 高质量头像重建、表情驱动和新视角渲染。不是上传任意一张照片后即刻瘦脸 | 当前排除；也与放弃 FLAME 的决定冲突 |
| LGM，ECCV 2024 Oral | 外部可输入单图或文字；ImageDream/MVDream 先生成四个视图，核心 U-Net 融合多视图 Gaussian；训练以 Objaverse 三维物体为主 | 生成通用三维资产，并非精确重塑原照轮廓。补出的背面和不同视角不是真实观察 | 当前不采用为修图底座 |
| LivePortrait，2024/2025 修订 | 单张源人像 + 驱动视频/图像/动作模板；官方界面也允许直接改姿态与表情滑块 | 有张嘴/闭嘴、微笑、眨眼、眉挑、视线、头部姿态；“眼睛张开比例”不等于眼球尺寸美颜；没有官方瘦脸、鼻梁、腰腿、衣物去皱参数 | 候选表情/姿态扩展；不能当全美颜替代品 |

DragGAN 官方 README 明确：“This GUI supports editing GAN-generated images. To edit a real image, you need to first perform GAN inversion using tools like PTI.” [官方说明](https://github.com/XingangPan/DragGAN/blob/336f120ce126aca6f55dc58537e76c10d19eabd0/README.md)。

FreeDrag 论文 §5.1 明确其扩散版本针对每张图训练 rank 16 LoRA，§5.6 指出二维点对可能把“抬头”误解释为“放大头部”。[论文](https://arxiv.org/html/2307.04684v4)。

FlashAvatar 官方页面明确写出输入为 monocular video sequence，pipeline 中将 Gaussian 嵌在 FLAME mesh；[官方项目页](https://ustc3dv.github.io/FlashAvatar/)、[官方代码说明](https://github.com/USTC3DV/FlashAvatar-code/blob/2d4cb25ca9166b795c03692aec12114d595eff15/README.md)。

LGM 论文 §3.2/§4.1 说明四个正交视图、ImageDream/MVDream 和白背景输入，§4.5 指出多视图不一致、生成视图仅 256×256 和大俯仰输入的局限。[论文](https://arxiv.org/html/2402.05054v1)。

LivePortrait 的真实滑块定义可在官方 [app.py](https://github.com/KlingAIResearch/LivePortrait/blob/9b294b3d0536135442ea73cb01e6cb3ca7029dd3/app.py#L102) 查到：eyes-open、lip-open、pitch/yaw/roll、位移、pouting/pursing/grin、smile/wink/eyebrow、eye gaze。这里没有将局部关键点约束映射为瘦脸等美容语义的模块。

## 二、性能数字的范围

| 项目 | 官方数字与硬件 | 数字不能证明的事项 |
| --- | --- | --- |
| DragGAN | 论文称多数编辑在单张 RTX 3090 上只需几秒 | 不含任意真实照片的 inversion 成本；不是 60ms 滑块刷新；CPU/MPS 虽有官方运行建议，未提供本机实测 |
| DragDiffusion | 论文附录 D：512×512 真实图，A100；LoRA 约 25s，latent 优化约 10–30s，最后去噪约 1–2s。README 更新称优化后训练约 20s；要求约 14GB 显存 | 20s 仅 LoRA 阶段，不能当整次修图延迟。缓存 LoRA 后仍需拖动优化/去噪；也不能外推浏览器 |
| FreeDrag | 核验的 README 和论文文本未给出可用于本项目的完整硬件/端到端耗时承诺；扩散代码明确使用 CUDA autocast、GradScaler 与 backward | 不能直接报“浏览器实时”或自行补一个速度数字；稳定性改进不等于端到端低延迟 |
| FlashAvatar | RTX 3090，已训练头像 512×512 渲染超过 300 FPS；从短视频训练重建为分钟量级 | 300 FPS 是已有 avatar 的渲染，不含采集、分割、FLAME 跟踪、逐身份训练；不是单张原图的编辑吞吐 |
| LGM | 论文宣称约 5s 生成三维内容；README 全部 ImageDream/MVDream/LGM 加载约需 10GB GPU 显存 | 已核验文本没有给出足够信息把“5s”外推到本机或浏览器；不要混淆前馈 Gaussian 重建与整个单图生成/网格转换。论文另称 Gaussian→NeRF→mesh 约 1min |
| LivePortrait | 论文摘要为 RTX 4090/PyTorch 12.8ms；仓库 speed.md 为 RTX 4090 + torch.compile 的各模块独立时间。官方提醒 Apple Silicon 可能慢约 20 倍，Intel 未测试 | 不是浏览器 WebGPU 数字；不包含图像上传、首次权重加载、全分辨率检测/裁剪/贴回和编码；不同文档的口径不能混算 |

LivePortrait 官方 [speed.md](https://github.com/KlingAIResearch/LivePortrait/blob/9b294b3d0536135442ea73cb01e6cb3ca7029dd3/assets/docs/speed.md) 给出主网络尺寸约 3.3 + 108 + 212 + 174 + 2.3 = 499.6 MB，辅助检测模型另计。量化、缓存和裁剪可以再研究，当前尚未实测这些改变对质量的影响。

DragDiffusion [论文附录 D](https://arxiv.org/html/2306.14435v6#A4) 与 [官方 README](https://github.com/Yujun-Shi/DragDiffusion/blob/ebe659a9c5b722f25d9690e74d813fca96531f97/README.md) 的 25s/20s 属不同版本表述，应同时保留出处。

## 三、浏览器 ONNX/WebGPU 核验

“开源 PyTorch 模型 → ONNX → ONNXRuntime WebGPU”不是已验证的完整路径。模型能导出、算子能执行、输出误差可接受和浏览器足够快，是四个独立问题。

1. DragGAN、DragDiffusion、FreeDrag 的编辑阶段不仅是固定网络前向；它们需要 latent 梯度、Adam 更新和循环优化，真图路径还涉及 inversion/LoRA。直接导出前向 ONNX 不会自动提供浏览器中的 backward、单图训练或完整算法。核验的官方仓库没有提供可直接接入本项目的浏览器 WebGPU 编辑实现。
2. FlashAvatar 和 LGM 的官方渲染使用 `diff_gaussian_rasterization` 自定义扩展，源码有 `device="cuda"`。浏览器可以另写 Gaussian 渲染器，但这只解决已有 Gaussian 的显示，不解决官方训练、生成和精确修图。不能用一个 WebGL splat viewer 证明整体算法已经迁到前端。
3. LivePortrait 主网络官方配置仍加载 `.pth`。仓库的 `arcface_onnx.py` 属 InsightFace 辅助识别网络，不等于人像生成主网络提供 ONNX/WebGPU。
4. LivePortrait WarpingNetwork 对 `[B,C,D,H,W]` 五维特征做 `F.grid_sample`；DenseMotion 也对五维 feature 做同类采样，并使用 `Conv3d`。核验的 ONNXRuntime WebGPU `GridSample` 明确限制 **only 4-D tensor is supported**。因此直接导出的 5D GridSample 没有现成对应的这一 WebGPU kernel。

关键源码证据：

- [LivePortrait warping_network.py](https://github.com/KlingAIResearch/LivePortrait/blob/9b294b3d0536135442ea73cb01e6cb3ca7029dd3/src/modules/warping_network.py)：`deform_input` 调用 `F.grid_sample`，输出形状注释 `Bx32x16x64x64`。
- [LivePortrait dense_motion.py](https://github.com/KlingAIResearch/LivePortrait/blob/9b294b3d0536135442ea73cb01e6cb3ca7029dd3/src/modules/dense_motion.py#L45)：`feature_repeat` 为 `(bs*(num_kp+1), c, d, h, w)`，第 50 行调用 `grid_sample`。
- [ONNXRuntime WebGPU grid-sample.ts 第 24–25 行](https://github.com/microsoft/onnxruntime/blob/469fb7e3a1643e728352482c801b74314430d3cf/js/web/lib/wasm/jsep/webgpu/ops/grid-sample.ts#L24)：`inputs[0].dims.length !== 4` 就抛异常。该文件最后核验的变更提交为 2025-03-20；2026-10-14 读取 main 时仍有此限制。
- [FlashAvatar Gaussian renderer](https://github.com/USTC3DV/FlashAvatar-code/blob/2d4cb25ca9166b795c03692aec12114d595eff15/gaussian_renderer/__init__.py)、[LGM gs.py](https://github.com/3DTopia/LGM/blob/fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c/core/gs.py)：自定义 Gaussian rasterizer 和 CUDA device。

LivePortrait 若做浏览器实验，需要先验证 5D 采样算子的改写或自定义 shader、其余网络算子、浏览器内存、冷/热耗时和输出误差。这里没有证明模型不可移植；结论只是**现成官方代码不能直接按所建议路径接入并承诺性能**。服务端 PyTorch 路径更接近官方实现，可与前端能力共享相同操作协议和输出合同。

## 四、身份、冻结区和轮廓精度

Drag 系列允许 editable mask，但其 mask 主要作用于 feature 或 latent 的损失项。DragGAN 对初始 feature 做 reconstruction loss；FreeDrag `functions.py` 第 216–218 行在非 mask 区域加 feature L1；FreeDrag 扩散分支第 210–212 行在 latent 上加 mask loss。这类软约束鼓励保持区域，并不能证明原始 RGBA 逐像素不变。[FreeDrag GAN 源码](https://github.com/LPengYang/FreeDrag/blob/a35d9882408a0405a6137a0dec13a308f5e5d3f7/functions.py#L216)、[扩散源码](https://github.com/LPengYang/FreeDrag/blob/a35d9882408a0405a6137a0dec13a308f5e5d3f7/FreeDrag--diffusion--version/utils/freedrag_utils.py#L210)。

LivePortrait 的 stitching 解决头像贴回时的肩部/裁剪对齐；其 `paste_back` 使用 `mask_ori * result + (1 - mask_ori) * img_ori`。模板软 mask 内部仍由生成结果影响，并非“冻结眼睛/耳朵/饰品”的语义权限系统。[官方贴回代码](https://github.com/KlingAIResearch/LivePortrait/blob/9b294b3d0536135442ea73cb01e6cb3ca7029dd3/src/utils/crop.py)。

FlashAvatar 的身份保持来自该身份视频的重建和训练；LGM 的图像一致性来自单图条件和生成视图。它们没有提供本项目的原照逐像素冻结、编辑后脸型几何误差或所有美颜控制的验收结论。

任何生成模型接入都应由本项目最后的合成器明确落实：允许修改区域之外逐像素复制原图；冻结眼睛/头发/耳朵/饰品的覆盖合同独立验证；同时检查边界连续性和身份。硬复制可以保证权限，但不能单独保证边界自然——允许区和冻结区的几何连接仍需修复。

## 五、对混合方案可采纳的部分

建议把基础修图组织成形变与补全两层，再扩展可选生成式编辑；三者保持同一份操作图、参数协议与 MCP 工具定义：

- **确定性局部形变**：脸部/五官/身形滑块以显式关键点、分区和可控几何场实现；核心要求是实际轮廓变化、冻区保护、无折叠、边界连续、背景直线和纹理保真。无生成模型时这一层仍完整可用。保直约束需要明确哪些背景线条受约束，不能只把“二维网格”当保直证明。
- **局部神经补全**：只对形变造成的露底、遮挡变化或被用户圈出的去人区域运行；用原照与冻结区作为条件，最终只写入允许区。模型主要补足原图没有的内容，而不是为每次美颜重画整张人像。局部窗口、批次缓存和最终高分辨率重建需要实测。
- **可选生成式编辑**：若需要大姿态/表情变化或自由拖动，再调用 LivePortrait/Drag 类工具；名称、执行耗时和图像重建行为应与基础滑块区分，并允许用户撤销。

这一方案可以吸收最新论文的约束求解、特征跟踪、身份条件或局部补全模块；不需要整体替换 Go + Rust/WASM + 浏览器识别的产品架构。MCP 是操作入口，不能消除底层算法和硬件限制，也不要求 WASM 内部运行一个 MCP 服务。

是否达到 8/10 仍要以原照测试决定。本研究没有给上述模型打产品效果分；目前也没有证据把官方论文样例质量直接转化成本项目每项功能已达标。至少应覆盖正负强度、多功能组合、侧脸/遮挡/多人、细发丝与饰品、直线背景、前端/服务端一致性和冷/热运行。

## 六、官方来源与版本

| 项目 | 论文 | 官方仓库核验提交 |
| --- | --- | --- |
| DragGAN | [2305.10973](https://arxiv.org/abs/2305.10973) | [336f120](https://github.com/XingangPan/DragGAN/tree/336f120ce126aca6f55dc58537e76c10d19eabd0) |
| DragDiffusion | [2306.14435](https://arxiv.org/abs/2306.14435) | [ebe659a](https://github.com/Yujun-Shi/DragDiffusion/tree/ebe659a9c5b722f25d9690e74d813fca96531f97) |
| FreeDrag | [2307.04684](https://arxiv.org/abs/2307.04684) | [a35d988](https://github.com/LPengYang/FreeDrag/tree/a35d9882408a0405a6137a0dec13a308f5e5d3f7) |
| FlashAvatar | [2312.02214](https://arxiv.org/abs/2312.02214) | [2d4cb25](https://github.com/USTC3DV/FlashAvatar-code/tree/2d4cb25ca9166b795c03692aec12114d595eff15) |
| LGM | [2402.05054](https://arxiv.org/abs/2402.05054) | [fe8d12c](https://github.com/3DTopia/LGM/tree/fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c) |
| LivePortrait | [2407.03168](https://arxiv.org/abs/2407.03168) | [9b294b3](https://github.com/KlingAIResearch/LivePortrait/tree/9b294b3d0536135442ea73cb01e6cb3ca7029dd3) |

初次使用的 `USTC3DV/FlashAvatar` 地址不存在；已由官方项目页核对实际为 `USTC3DV/FlashAvatar-code`。LGM README 文件名为小写 `readme.md`。这些纠正避免把错误链接或不同项目的指标混用。

## 七、2024–2026 较新候选的复核

这批项目更接近“局部二维形变 + 补全”方向，优先级高于把头像/三维生成模型硬塞进美颜主链路：

| 项目 | 适用能力 | 真实接入条件与硬限制 | 结论 |
| --- | --- | --- | --- |
| **Inpaint4Drag**，ICCV 2025 | 直接在像素空间做双向 warp，输出 warped preview，再调用任意 inpainting 模型补露底和边界；官方报告 512² warp 0.01s、inpaint 0.3s（Tesla V100） | Python/PyTorch、`diffusers 0.35.2`、`transformers 4.57.1`；默认 SD1.5 inpaint + TAESD + LCM LoRA；需 deformable mask。已独立跑六档官方warp+现有LaMa基线 | **原样组合已拒绝进入生产**：鼻口联动、旧皮肤残留、LaMa再生旧轮廓。只保留分层接口研究；未运行论文扩散补全，不把0.3s当WebGPU指标 |
| **DirectDrag**，WACV 2026 | mask-free、prompt-free drag；Auto Soft Mask + Readout-Guided Feature Alignment + latent warpage，允许缓存每图 LoRA | 官方 README 要求 NVIDIA CUDA、建议 ≥16GB VRAM；SD1.5/VAE 首次下载约 6GB；PyTorch 2.0.1、diffusers 0.24.0；README明确大于 512px 会缩放。没有浏览器导出或 CPU 方案 | **研究备用**。交互更省 mask，但“自动可编辑区”不适合对眼睛/头发/饰品做严格权限；先和 Inpaint4Drag 对同一组照片比边界与身份 |
| **InstantDrag**，SIGGRAPH Asia 2024 | 面向真实 talking-head 视频训练的 FlowGen/FlowDiffusion；细粒度脸部点拖动，config-3 对多点更局部 | CUDA、Torch 2.2.2+cu121、diffusers 0.30.1；大动作可能超出训练光流范围；官方免责声明称非脸部场景不保证 | **可作脸部点拖动实验**，不是全身/衣服/背景方案；必须验证侧脸、遮挡和冻区 |
| **AGGN**，WACV 2025 | 结构感知 human body reshaping；AAG 捕获身体部位关系，BSD 改善纹理，BR-5K 上报告 SSIM/PSNR/LPIPS | README要求 RTMDet/RTMPose 和 PWCNet/CuPy 训练数据；核验仓库有 train.py，但 **没有 README 引用的 evaluate.py，也未找到发布的身形网络权重**。当前不是完整可复现推理包 | **方法研究候选，暂不能直接接入**。任务贴近“腰、腿、身体重塑”，但缺少推理入口/权重；不要仅凭论文指标替换当前几何场 |
| **ZIM**，ICCV 2025 Highlight | zero-shot interactive image matting，点/框提示输出细粒度 alpha，面向发丝和微小边界 | 官方CPU路径、vit-b约380MB权重已实际执行五组提示×四alpha；原照编码约8–9s，缓存decoder约0.48–0.91s | **原始alpha不接受为可动面部physical**：耳负点失败，加密组原耳区90.95%仍alpha≥0.5；可保留抠图研究，详见[评审](../qa/zim-20261014/report.md)。不是形变/去人生成器，未验证浏览器 |
| **MEMatte**，AAAI 2025 | 高分辨率 memory-efficient **trimap-conditioned** matting；adaptive token routing，提供 AIM-500/Composition checkpoints | PyTorch/detectron2；inference.py明确读取trimap，并把确定背景/前景钳到0/1；还需 `max-number-token` 和公开 Google Drive checkpoint。没有现成浏览器 ONNX 路径 | **可作服务端高分辨率 matting 对照**，需已有分割或画笔产trimap；不承担身体重塑 |
| **MP-Mat**，ICLR 2025 | 3D/instance-aware human matting and editing with multiplane representation | 官方 README 核心内容仍写“代码将组织后开源”，当前接口/权重/运行链不完整 | **暂不接入**，避免把论文标题当可复现实现 |

Inpaint4Drag 官方资料：[仓库](https://github.com/Visual-AI/Inpaint4Drag/tree/bfb0fafd456ac01ee23ade0cc04b64c9781eaa87)、[论文](https://arxiv.org/html/2509.04582)。论文把拖动明确分成双向像素 warp、露底/轮廓 dilation mask、标准 inpainting；其限制仍是 mask 与点对的解释依赖用户，且最后内容由生成模型补出。它最符合本项目提出的“二维网格保结构 + 局部神经补全”分层。

DirectDrag 官方 [README](https://github.com/frakw/DirectDrag/tree/9a0c32fc916a3a142a74f537cfb701cc1ce89dfa) 明确首次运行会做每图 LoRA，后续可缓存；输入长边大于 512 会自动缩放，默认需要至少 16GB NVIDIA 显存。它适合异步生成式能力，不适合作为滑块每帧底层。

AGGN 官方 [README](https://github.com/Randle-Github/AGGN/tree/312176f9cbada0b0b263c32fa421539c721eb536) 给出 BR-5K 数据布局和指标：作者报告 Ours SSIM 0.8427、PSNR 26.4100、LPIPS 0.0643；这些是在其数据集和训练配置上的结果，不是本项目人像照片的质量分。检测模型和 PWCNet/CuPy 数据管线也说明它是研究训练代码，不是拿来即用的 Go/WASM 库。

ZIM [固定源码](https://github.com/naver-ai/ZIM/tree/392e613c8a3d395ce4dda3b51edb6dabc1fd0004) 和 [HF vit-b 模型目录](https://huggingface.co/naver-iv/zim-anything-vitb/tree/main/zim_vit_b_2043) 提供encoder/decoder ONNX；encoder输入image，输出image_embeddings、feat_D0/1/2，decoder输入这些缓存feature与point_coords/point_labels/attn_mask。因此可用原图一次编码、多个主体/点提示复用feature；不需要声称全模型每帧重算。MEMatte [inference.py](https://github.com/linyiheng123/MEMatte/blob/main/inference.py) 明确要求trimap，不能称trimap-free。

Inpaint4Drag也需要二次封装：[固定warp源码](https://github.com/Visual-AI/Inpaint4Drag/blob/bfb0fafd456ac01ee23ade0cc04b64c9781eaa87/utils/drag.py) 使用top-4 inverse-distance displacement、整数round；逆向reference超过100会均匀抽样；`cv2.RETR_EXTERNAL`加实心外轮廓会吞掉mask内部眼睛等洞。它没有Jacobian/保直约束，默认inpaint会扩张轮廓带；本项目仍必须加独立冻区、抗锯齿和几何验收。采纳分层思路不等于原样接入就达到最优美颜。

## 八、更新后的实施排序

1. **Inpaint4Drag 可替换实验已完成并归档负例**：原照六档官方warp+现有LaMa不通过自然轮廓与五官保持，拒绝直接接生产。输入/输出协议继续可借鉴；本次没有移植Rust版本，也没有复现论文扩散补全。实际结果见 [评审](../qa/inpaint4drag-20261014/report.md)。
2. **把 ZIM 或 MEMatte 放在 mask/alpha 侧**，只负责主体边界和发丝/衣物软边；用这层生成的 alpha 约束允许区，不能把 matting 模型当成几何变形模型。
3. **局部异步补全先复用已有 LaMa 服务**，再以 Inpaint4Drag 的 SD1.5+TAESD+LCM 作为 GPU 对照。补全结果必须经过合成器的冻结区复制和边界 QA；没有 AI 时保留确定性 patch/邻域填充。
4. **身形研究 AGGN/FBBR 系列**：AGGN当前缺发布推理链，不阻塞现有实现；先用其结构约束作为参考。任何获得真实权重的候选再按身体mask、分辨率、CPU/GPU时间和衣物纹理验收，禁止未经验证合并到脸部field。
5. **LivePortrait/InstantDrag/DirectDrag 做可选“脸部拖动/表情”实验**，只在服务端异步运行；暂不承诺浏览器端。真实照片必须记录模型初始化、单次编辑和缓存编辑的冷/热耗时。

推荐的评估不是只看 LPIPS：每个功能至少同时记录目标关键点误差、轮廓 Chamfer/边界 IoU、冻结区像素差、背景直线弯曲度、发丝 alpha SAD/梯度误差、身份 embedding 相似度、冷/热耗时和显存。Inpaint4Drag 论文也提醒 LPIPS 会把“正确的形变”惩罚为差异，因此不能单独用它评估瘦脸或身形。
