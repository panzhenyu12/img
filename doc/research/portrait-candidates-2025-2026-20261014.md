# 2025–2026 人像修图候选：官方代码、模型与实际可接入性

核验日期 2026-10-14。目标是提升脸/身形真实轮廓、源 alpha 与露底补全，不用论文标题或作者样例给本项目打 8 分。本次明确覆盖 Inpaint4Drag、ZIM、MEMatte、AGGN、DirectDrag；它不是整个领域的穷尽排名，也没有证据证明某个候选是全球最优。判断分别标明官方信息和本机实测。

## 采用结论

| 候选 | 负责的环节 | 官方发布状态 | 本项目状态与决定 |
| --- | --- | --- | --- |
| Inpaint4Drag / ICCV 2025 | 用户可动 mask + 点对的双向像素 warp，露底/轮廓带补全 | 代码公开；默认 SD1.5 inpaint + TAESD + LCM；无需训练该 warp | 已在用户原照跑六档官方 warp + 现有 LaMa 基线；**拒绝这套原样组合进入生产**，保留分层接口研究 |
| ZIM / ICCV 2025 Highlight | 类别无关交互抠图/软 alpha | 官方 vit-b encoder/decoder ONNX 公开，不需登录；源码默认 CPUExecutionProvider | 已实跑五组提示×四alpha，约8–9s编码/0.48–0.91s缓存decode；耳负点失败，**原始alpha不接受为可动面部physical**，保留交互抠图研究 |
| MEMatte / AAAI 2025 | 高分辨率、由 trimap 约束的 alpha | 公开代码与 Google Drive checkpoint；PyTorch + detectron2，需原图和 trimap | 官方链核对；**可作为后续高分辨率对照，未实跑**。已有语义 mask/手工笔刷需二次封装为 trimap |
| AGGN / WACV 2025 | 面向身体重塑的结构关系与高频外观 | train.py 与模型代码公开；核验完整 tree 未见 `evaluate.py`、身形网络 checkpoint | **只研究方法，不直接依赖**；没有可复现预训练推理，不以 RTMDet/RTMPose 权重冒充重塑模型 |
| DirectDrag / WACV 2026 | 自动 soft mask + readout guidance + latent warpage 的生成式点拖动 | 代码、readout weights、样例公开；另需 SD/VAE 和每图 LoRA | **可选 GPU 异步工具研究，当前未实跑**；不进入每帧美颜滑块，不承诺 CPU/浏览器路径 |

没有一个上述模型同时提供“瘦脸、瘦腰、腿形、衣物去皱、背景不变”的现成完整控制。alpha、几何场、补全各自要独立验收，不能把 matting 输出存在解释成轮廓重塑成功。

## 1. Inpaint4Drag：接口可用，原样几何不通过

官方 [论文 2509.04582](https://arxiv.org/abs/2509.04582) 与[固定仓库](https://github.com/Visual-AI/Inpaint4Drag/tree/bfb0fafd456ac01ee23ade0cc04b64c9781eaa87) 把拖动分为 bidirectional warp 与任意 inpainting 补全。作者 512² 的 warp 0.01s / inpaint 0.3s 是论文 GPU 实验数字，不是本项目 CPU、完整 UI 或浏览器时间。

[固定 warp](https://github.com/Visual-AI/Inpaint4Drag/blob/bfb0fafd456ac01ee23ade0cc04b64c9781eaa87/utils/drag.py) 是 top-4 逆距离位移 + round 整数坐标，逆参考点最多 100；外轮廓填充会吞掉眼睛等内部 mask 洞，没有本产品要求的连续 Jacobian/无折叠、内器官冻结或背景保直保证。

本项目原照 `0、±30、±60、+100` 的隔离实验：[完整负例评审](../qa/inpaint4drag-20261014/report.md)。非零 CPU warp 137–223ms；目标 mask IoU 0.998–0.999，但鼻口明显联动，+100 眼区变动 167 像素；LaMa 3.5–6.1s 补全把皮肤延伸回旧轮廓。源/目标 mask 正确无法证明最终自然边缘正确。官方 zero warp 确实为 identity，却仍输出 8,027 像素轮廓补全带，调用层应对 0 参数直接跳过补全。

可采用：可替换的 warp/露底/轮廓补全接口、GPU 补全的独立层。需要二次封装：冻结合成器、语义权限、真实几何约束与 QA。拒绝直接依赖：未修改官方 warp 作为所有美颜底层；本次未跑论文默认扩散补全，不夸大成整篇论文复现或否定。

## 2. ZIM：原图一次编码，交互 decoder 可缓存

官方 [论文 2411.00626](https://arxiv.org/abs/2411.00626)、[固定源码](https://github.com/naver-ai/ZIM/tree/392e613c8a3d395ce4dda3b51edb6dabc1fd0004)、[官方 HF revision](https://huggingface.co/naver-iv/zim-anything-vitb/tree/883e9bbb15ca35a6afe16b72ae7e55cc13efbc78/zim_vit_b_2043)。公开权重实得：

| 文件 | 字节 | 实际 SHA256，与官方 LFS 一致 |
| --- | ---: | --- |
| encoder.onnx | 360,680,416 | `6123ea10722dd43a0f8bbb96cd6a0532d8a65e2b1e8339f0cd6593d494807809` |
| decoder.onnx | 19,330,176 | `a99d14bac3dc28b17809020b1711b1991bd5b7eb18f678c3624ce508748cfa11` |

官方 encoder 输入 RGB，长边缩放到 1024，均值/std 归一化后右/底 padding；输出 image_embeddings 与 feat_D0/1/2。decoder 接点/框和 attention mask，points 追加 `[-0.5,-0.5] / label -1`；框用 label 2/3；四个候选 logits 经原照尺度插值与 sigmoid 得 alpha。不能漏掉 preprocess、prompt padding 或把 logits 当 alpha。

[encoder.py](https://github.com/naver-ai/ZIM/blob/392e613c8a3d395ce4dda3b51edb6dabc1fd0004/zim_anything/modeling/encoder.py) 与 [decoder.py](https://github.com/naver-ai/ZIM/blob/392e613c8a3d395ce4dda3b51edb6dabc1fd0004/zim_anything/modeling/decoder.py) 默认 CPUExecutionProvider，可调用 CUDA，但默认 requirements 写 `onnxruntime-gpu`，当前 CPU 实验以隔离 ONNXRuntime CPU 1.19.2 代替它，不能误报官方包直接兼容本机。原图 feature 可缓存，提示变化只重算 decoder。

本实验在 `experiments/zim-20261014/`：未修改官方 predictor/model/preprocess；只将线程设为 3，并跳过与推理无关的包初始化依赖。正点来自原照脸/皮肤/鼻/眉/唇/眼检测，负点来自耳/发/颈/饰品；保留原始四候选和官方 argmax，不靠语义裁剪掩盖整头/整人选择。与 BiSeNet coverage/归档 physical alpha 的 IoU/差异只表示 disagreement，因为两者都不是 alpha ground truth。

本机实测已完成并冻结，见 [ZIM原照评审](../qa/zim-20261014/report.md)。五组×四候选中的加密正负提示最高score输出仍有90.95%原耳区alpha≥0.5，耳负点alpha0.981/0.9997；发基本排除，颈内部仍0.99%残留。原始alpha不能直接作为可动脸物理层；语义排除适配有耳缘细残留，未验收。原照编码7.97/9.02s，缓存decoder+postprocess0.48–0.91s，进程峰值RSS5,533MiB；这不是浏览器时间。需要二次封装：主体/部位语义选择、允许/冻结区、缓存生命周期、边缘权限与内存核验。浏览器 ONNX/WebGPU 尚未运行，不能根据 ONNX 后缀承诺前端实时。

## 3. MEMatte：必须提供 trimap

官方 [论文 2412.10702](https://arxiv.org/abs/2412.10702)、[固定 README](https://github.com/linyiheng123/MEMatte/blob/3c887e2a517b38f936b97f27d933c5897c6d47a1/README.md)、[固定 inference.py](https://github.com/linyiheng123/MEMatte/blob/3c887e2a517b38f936b97f27d933c5897c6d47a1/inference.py)。代码实际从 `trimap/` 或 `trimaps/` 读图，sample 传 `trimap`，输出再把 trimap==0/1 钳为 0/1；因此 **不是 trimap-free**。

memory-efficient 来自 adaptive token routing/LTRM 与 max-number-token 控制，不表示无限分辨率或 CPU 实时。官方 Composition-1k/AIM-500 的 SAD/MSE/Grad/Conn 是模型评测，不可转换为本产品美颜分数。源码依赖 detectron2 配置/instantiate/checkpoint，默认训练 device；未提供官方浏览器 ONNX 导出。后续可用当前分割生成 foreground/background/unknown trimap，独立核验高分辨率边缘；需要识别过硬错误 trimap 会被模型原样保留的局限。本次没有下载 checkpoint、没有本机耗时或输出证据。

## 4. AGGN：任务相关，发布的推理链不完整

官方 [论文 2404.13983](https://arxiv.org/abs/2404.13983)、[固定仓库](https://github.com/Randle-Github/AGGN/tree/312176f9cbada0b0b263c32fa421539c721eb536)。AAG 对身体部位关系建图、BSD 强化高频纹理，任务与身形重塑相符。官方 README 报 BR-5K SSIM 0.8427 / PSNR 26.4100 / LPIPS 0.0643，这是作者研究数据，不是本项目腰腿效果结论。

核验完整 Git tree（未截断）：有 `train.py` / `model/aagn.py`，没有 README 引用的根目录 `evaluate.py`，未见 `.pth/.pt/.ckpt/.onnx` 身形 checkpoint。README 给的是 RTMDet/RTMPose 检测权重及 PWCNet/CuPy 数据生成链，不能把这些当 AGGN 预训练重塑网络。本次未训练、未获得官方重塑权重，不虚构 runnable 推理。可借鉴部位间结构一致性与衣物高频纹理检查；不要直接加依赖并承诺质量。

## 5. DirectDrag：2026 的生成式备用路线

官方 [论文 2512.03981](https://arxiv.org/abs/2512.03981)、[固定 README](https://github.com/frakw/DirectDrag/blob/9a0c32fc916a3a142a74f537cfb701cc1ce89dfa/README.md)。自动 soft mask、readout-guided feature alignment 与 latent warpage 减少点拖动的 mask/text 操作。但首次真实照片编辑仍训练每图 LoRA，之后缓存；latent 优化不等于固定前向网络导出。

README 明确 Linux/NVIDIA CUDA、建议 ≥16GB VRAM；约 1GB repo/readout/sample 与额外约 6GB SD/VAE cache，长边 >512 会缩小。源码要求老版 Torch2.0.1/diffusers0.24.0，无法与生产 CPU worker 混装。需要独立 GPU 环境、原照哈希和每图模型缓存、安全最终合成、逐档目标误差/身份/细节核验。官方没有已验证 CPU/WebGPU 编辑路径；不以当前 Intel Mac 勉强执行作为正常性能承诺。

官方可选 GScore 调 Gemini API，只是 LLM-as-judge，不代表已客观到 8 分；IF=1−LPIPS 与 DIFT MD 也不验证原照冻结合同。不开账户、不调用付费评分。没有本机运行结果，保持 research-only。

## 产品接入合同

确定性手工功能必须独立运行。Rust/WASM 接受明确语义 mask、局部 field/点对并输出位移、目标 alpha、露底和允许写入范围；Go/MCP 调用同一操作协议；模型服务只提供可选 alpha 或补全结果。最后合成器复制冻结像素，另检查耳根/发际/颈部连接。

接入前要求：0 参数 identity；正负多档真实轮廓运动；独立解剖部位副作用；数值 Jacobian/折叠；背景线与衣物纹理；身份与细节；冷/热耗时与内存；客户端/服务端一致性。更好的 source alpha **不能修复旧变形场从 alpha=1 到0的位移断裂**，不得把 matting 实验成功写成脸型算法已达标。FLAME 继续排除。
