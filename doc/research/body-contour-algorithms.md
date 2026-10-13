# 身体轮廓算法与人体解析取舍

日期：2026-10-13。以真实腰/臂/腿轮廓改变、固定背景及非目标部位为验收标准；不以位移数量证明自然度。当前核心实现和限制见 [FUNC-012](../func/FUNC-012-body-semantic-masks.md)。

## 论文与当前实现的关系

[Structure-Aware Flow Generation for Human Body Reshaping](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Structure-Aware_Flow_Generation_for_Human_Body_Reshaping_CVPR_2022_paper.html)，Ren 等，CVPR 2022，DOI `10.1109/CVPR52688.2022.00760`，提出骨骼/Part Affinity Fields 条件下的结构感知 flow 和 compositional attention，提供 BR-5K 配对训练数据、[官方代码](https://github.com/JianqiangRen/FlowBasedBodyReshaping)及 [预训练权重](https://drive.google.com/file/d/1E2kwYVCopNzcmi2U2vWXORyiJmExcezD/view)。[arXiv 原文](https://arxiv.org/abs/2203.04670)。BR-5K 需要向作者申请，当前未发邮件申请。

当前实现采用骨轴、局部 flow 与独立背景的架构思想，自行实现左右真实 alpha 截面和一维单调逆映射；**没有运行 SAFlow 预训练模型、复现其训练、attention 或论文指标**。SAFlow 对完整人体宽度调节有研究价值，不能直接证明独立前臂、衣服去褶皱、腿长或背景完全固定。若接入权重，先保留其官方 Pose/PAF 输入与预处理，再对照当前同一批实拍和每部位固定像素要求，不能替换模型名字就宣称论文算法已完成。

[Fast Multi-Level Foreground Estimation](https://arxiv.org/abs/2006.14970)，Germer 等，2020，公开 [PyMatting](https://github.com/pymatting/pymatting)。当前共享 Rust 模块 `face_matting.rs` 复现多尺度 F/B 颜色估计，以已有 alpha 解软边前景/背景颜色。它改善混合色拖影，**不估计 alpha，不会自动修正错分的衣袖或包**。身体使用此模块，纯前景/纯背景仍以源图锚定。

## 可采用项、需要封装项与不建议直接依赖项

| 项目 | 分类及依据 | 当前产品状态 |
|---|---|---|
| MediaPipe Pose Landmarker | 可采用；33 点、每个人主体分割、浏览器本地推理。需把 confidence 与真正 matting 区分 | 已接入；可见性和裙下/交叠部位有保守限制 |
| 像素骨轴与真实双侧截面 | 可采用；处理照片长宽比、非对称外边和宽衣 ROI 漏边 | Rust/WASM 已实现，真实图仍需自然度复核 |
| PyMatting F/B 多尺度估计 | 需要封装；只负责颜色分离，与主体/部位 alpha 和固定锚点结合 | 已有 Rust 共享复现，透明原 RGBA 走自己的 alpha |
| SAFlow / FlowBasedBodyReshaping | 需要二次封装及对照；官方权重、Pose/PAF 和 flow 输出不能混用当前 Pose 格式 | 仅研究，尚未模型复现 |
| SCHP | 需要二次封装；LIP/ATR 能解析衣物及左右肢体，但预处理、模型版本和遮挡仍要验实拍 | 候选，尚未采用产品依赖 |
| FASHN Human Parser | 需要二次封装；有脸、发、衣物、手、臂、腿等 18 类；重模型，可先本地异步实验 | 已下载真实模型、跑五张及导出检查 ONNX，未接 UI/自动 partMask |
| 社区 humanparsing-onnx | 不建议直接依赖未知导出；需核实来源、输入格式和与官方输出一致性，两个模型合计约 534MB | 仅调查 |
| matei-dorian SegFormer-B0 human parsing | 不建议用于当前身形；公开模型卡 mIoU 0.0726，左右臂 IoU 0 / 0.0001 | 已拒绝，不因为 15MB 就接入 |
| 全图统一 warp、Pose polygon 当主体 alpha | 不建议；骨点支持范围不能隔离实际人物，正是背景动而轮廓不动的来源 | 已删除无 subjectMask 的 body 兼容路径 |

## 人体解析模型实验

### SCHP

论文 [Self-Correction for Human Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)，TPAMI 2020，DOI `10.1109/TPAMI.2020.3048039`。LIP 20 类/ATR 18 类与左右臂腿可给 partMask 提供依据，仍要结合 Pose 进行实例归属和可见性判断。

社区 [Longcat2957/humanparsing-onnx](https://huggingface.co/Longcat2957/humanparsing-onnx)，调查版本 `ac3450d03ab21407f01e5cb1711ba42201cf30d6`，两个模型合计约 534MB。当前未核实完整预处理及 official/ONNX 数值一致性，因此不作为已可用依赖。

### FASHN Human Parser

[官方代码](https://github.com/fashn-AI/fashn-human-parser) / [模型](https://huggingface.co/fashn-ai/fashn-human-parser)，实测版本 `1f80c34dbab321c5730dda5c3fea279fd3e97498`，SegFormer-B4，约 244MB。类别：background、face、hair、top、dress、skirt、pants、belt、bag、hat、scarf、glasses、arms、hands、legs、feet、torso、jewelry。

官方推理采用 OpenCV INTER_AREA 缩放至 width 384 × height 576、RGB/255、ImageNet mean/std。已导出 opset17 ONNX 并通过 checker，input `pixel_values=[1,3,576,384]`，output `logits=[1,18,144,96]`，ONNX 约 245MB。官方代码与 HF processor 的缩放方式有差异，最终接入必须锁定一致预处理；目前没有浏览器推理及 official/ONNX 一致性验收。

五张实拍 PyTorch CPU 实验约 8.77–44.97 秒/张，期间与其它模型验证并行竞争 CPU，不能当作稳定性能基准。结果和类概率保存在 `.run/fashn-results/`，模型与依赖在忽略的 `.run` 下隔离，不改变背景 worker 的固定 torch/numpy 环境。

模型能够把包、手、头发从主体中分类，但宽袖和躯干仍同属 top，衣裤覆盖的肢体不独立。**有衣物 parsing 也不能直接宣称目标部位准确或身形自然度达到 8 分。**下一步需 Pose 实例归属、服装连通域和交叠部位保护，再输出可信 partMask；失败部位保持手工修正入口。

### 轻量 B0 拒绝依据

[模型卡](https://huggingface.co/matei-dorian/segformer-b0-finetuned-human-parsing)，版本 `6eb347434a054af8ff79ec03e8596bc3742d8ac9`，约 3.7M 参数/15MB。其公开总体与臂部指标不足以承担目标部位分离，本轮未采用。

## 后续实验顺序

先对当前真实轮廓 15 项回放逐项保存失败，按袖子/包误动、alpha 错边、窄带结构线及纹理重建分类。接下来分别验证真 partMask 和更精确边缘，不将衣物同类区域粗暴当作整条肢体；再运行 SAFlow 官方权重，检查相比当前确定性路径是否真正改善质量及速度。身高、腿长、体态和衣物大褶皱保留独立算法任务，不能伪装成宽度调节。
