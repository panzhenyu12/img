# 人像与衣物修图算法调研

更新时间：2026-10-13

这份调研用于确定一期 Demo 到后续 AI/高质量版本的算法边界。当前一期以可回放、无 AI 也可用为前提；论文只作为算法方向和评测依据，不代表仓库已经包含对应模型。

## 论文与适用环节

| 方向 | 论文/资料 | 对本项目的启发 | 落地阶段 |
|---|---|---|---|
| 人像体型参数化 | [Parametric Reshaping of Portrait Images for Weight-change](https://doi.org/10.1109/mcg.2018.011461529) | 用轮廓参数和保护区域表达瘦身、腰部和体型变化，避免全图缩放 | 一期语义 polygon + 局部 warp；二期参数化人体网格 |
| 视频人像轮廓 | [Parametric Reshaping of Portraits in Videos](https://doi.org/10.1145/3474085.3475334) | 轮廓控制点、时序稳定和边界连续性，适合未来视频客户端 | 二期视频/时序 |
| 人体解析与体型 | [Shape Customisation of Human Subjects Based on Human Parsing Technology](https://doi.org/10.15221/17.288) | 先得到人体部位，再分别处理腰、胯、腿、手臂，背景作为保护区 | 一期 Pose 部位 polygon；二期 parsing mask |
| 人像修图数据 | [PPR10K](https://doi.org/10.1109/cvpr46437.2021.00071) | 人脸/人体 mask、局部 retouch 标签、组级一致性和质量评测 | 二期训练数据与回归集 |
| 像素自适应修图 | [Learning pixel-adaptive weights for portrait photo retouching](https://doi.org/10.1016/j.patcog.2023.109775) | 按像素预测效果权重，解决统一强度导致的眼睛、头发和衣物被误伤 | 二期轻量模型 |
| 实时人像美化 | [Real-time portrait image retouching extended from DualBLN](https://doi.org/10.1016/j.eswa.2023.121958) | 轻量网络与实时推理；可作为 WebGPU/移动端模型基线 | 二期 GPU/移动端 |
| 拖拽式局部编辑 | [Inpaint4Drag](https://doi.org/10.1109/iccv51701.2025.01701) | 拖拽控制点、双向 warp 与 inpainting 联动，解决轮廓移动后的窄环补全 | 二期 AI/生成式修复 |
| 衣物与人体联动 | [Body Shape-Aware Object-Level Outfit Completion](https://doi.org/10.3390/app13053214) | 衣物区域、人体轮廓和服装形状需要联合建模，不能只平移矩形 | 二期服装重塑 |
| 磨皮与保边 | [Self-quotient Referential Bilateral Filter](https://doi.org/10.17706/ijcee.2015.v7.875) | 低频平滑与边缘/纹理保护分开处理，适合皮肤和布料轻度平整 | 一期 Rust 保边/频率分离 |
| 形变边界优化 | [Shape-optimizing hybrid warping for image stitching](https://doi.org/10.1109/icme.2016.7552928) | 通过固定边界和优化过渡区减少拉伸接缝 | 一期 RBF feather；二期网格/MLS/TPS |
| 生成式人脸美化 | [Applied Research on Face Image Beautification Based on a Generative Adversarial Network](https://doi.org/10.3390/electronics13234780) | 适合 AI 一键美化和复杂补全，但需要模型、GPU 和输入质量控制 | 二期可选 AI |

## 一期采用的算法基线

1. **脸型与五官**：MediaPipe Face Landmarker 提供归一化点；Web 保存 polygon mask 和最多 128 个点。Rust 使用局部高斯 RBF 逆采样；`faceWidth` 在没有目标点时使用中心保护、双侧轮廓带和边界渐隐模板。选区外通过 `mask::composite` 保持原像素。
2. **身体、腰、腿和手臂**：Pose 33 点派生躯干、腰、胯、左右上臂/前臂/大腿/小腿 polygon，并把 `bodyPart`、源 landmarks 写进 `body.adjust`。这比整个人体 bbox 安全，但仍不是像素级人体 parsing。
3. **衣物去褶皱**：`clothing.adjust` 的 `smooth` 以局部模糊低频平面压低连续褶皱，`texture` 按高频残差保留织物纹理；通过局部 polygon/套索蒙版限制作用范围。严重褶皱和缺失纹理不伪装成传统平滑可解决的问题。
4. **手工降级**：没有模型、模型加载失败或 AI 不可用时，矩形、椭圆、套索、多边形、液化、修复和仿制图章仍可创建同一 EditGraph 操作。

## 需要二次封装的开源项目

| 项目 | 采用判断 | 封装要求 |
|---|---|---|
| MediaPipe Tasks Vision | 可采用 | 模型缓存、坐标归一化、Face/ Pose 语义输出、失败回退；不能把 bbox 当分割 mask |
| `rust_mls` / TPS / Delaunay | 候选 | 统一成 `WarpField` 接口，固定 anchor、边界 feather、CPU/WASM/native 一致 |
| `photon-rs` | 候选 | 只复用成熟滤镜算子，仍由 Operation 管理蒙版、强度、撤销和随机种子 |
| ONNX Runtime/Tract | 后续 | 明确模型版本、输入尺寸、WebWorker/GPU provider 和失败时手工回退 |
| Inpainting/扩散模型 | 后续 | 独立 AI Operation，保留原图和 mask；模型失败不阻塞传统修复 |

## 质量验收指标

- 轮廓参数为 0 时输出逐像素等于输入。
- 轮廓调整只改变语义 polygon 内，mask 外背景差分为 0（羽化带按容差统计）。
- 只调 `faceWidth` 时眼睛、鼻子、嘴的控制点中心位移小于脸宽 1%。
- 正负方向对称，中心保护区不发生整体缩放。
- 衣物 `texture=100` 的高频差异不低于低纹理设置，alpha 不变。
- 同一 EditGraph 的 WASM/native 输出逐像素一致。
- 检测、AI 或模型不可用时，手工操作和工程回放仍可完成。

## 当前限制与下一步

一期没有真实人体 parsing、衣物分割、三维脸网格、GPU renderer 或生成式模型。二期优先顺序是：人体 parsing 与窄环补全、五官 landmark 子集的参数化目标点、MLS/TPS/三角网格 WarpField、衣物纹理感知 inpainting，最后再接视频时序和云端 AI 工作流。
