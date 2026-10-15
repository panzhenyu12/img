# 技术选型与开发文档（Web Demo）

近期候选的可采用、二次封装与不直接依赖项见[2025–2026官方评审](research/portrait-candidates-2025-2026-20261014.md)。本项目算法研发见[实现记录](research/portrait-flow-development-20261014.md)：五官V4b与脸型v19已接共享核心，工程接入不等于自然度通过；拒绝的隔离版本继续保留。

2026-10-15 接入勘误：ZIM 官方 ONNX 包装器同时收到框和点时，会覆盖点 token；旧组合提示实验不能据此判定负点无效。独立适配器已实跑拼接点/框 token 并保留 attention 消融；原始互斥 box/points 实验不受影响。ZIM 继续属于需要二次封装与画质验收的辅助层，未替换生产检测。源颜色分解、局部补全与前后遮挡的新合同见 [FUNC-023](func/FUNC-023-portrait-layer-matting.md)。

2026-10-14追加生产集成：v19联合脸颈已按用户授权接入native/WASM，使用Rust RBF/SVF与Clarabel固定QP。可信脸/上颈共同传输、低颈静止，缺可靠连通保留face-only/手工回退；耳/发/配饰和保护保持。chin/forehead跨零不切换QP，non-normal合成也遵守冻结/保护，无新增公开实验参数或AI依赖。四档原照UI/Go/native/WASM与旧candidate全RGBA一致，受影响10项E2E、Go/Web通过；最终gate源码face42项通过，稍早全Rust206+10仅属gate微调前快照。v14–v18/颈层补全仍拒绝。**用户随后明确拒绝v19瘦脸的审美效果，认为不自然、曲线不好看；接入验证通过不等于质量验收。** 脸、五官与身体曲线需要各自优化，不把统一速度框架当作统一审美规则。状态与耳侧12px原skin岛/硬边、单原照限制见[生产集成](qa/face-joint-v19-integration/review.md)，不升8分。

2026-10-14路线修订：FLAME/DECA按用户要求放弃。采用前景分层、连续形变、物理coverage传输与材质局部补全研究方向；保持发/耳/饰品/用户冻结。v19已工程接入，但新瘦脸审美仍被用户拒绝，不承诺消灭所有裂缝或达到8分。Gemini全图网格+LaMa仅采纳双阶段架构，不照搬零裂缝/60FPS/删除分层承诺。比较见[混合路线](research/hybrid-portrait-route-20261014.md)与[建议评审](research/gemini-algorithm-review-20261014.md)。

此前五官V4b修复眉眼串扰、边界/跨零、眼倾斜中心放大和identity透明RGBA，195+10与相应双端对账是该快照的历史记录。它不替代脸型/身形的具体曲线设计，五官自然度仍未8分，见[FUNC-020](func/FUNC-020-feature-svf-quality.md)。人脸v19现已独立集成，旧失败版本不因共享连续流框架而被追认为通过。

2026-10-13 更新：人脸增加局部颜色线 alpha 校正与颈部底层 donor；皮肤增加多尺度细节与紧凑瑕疵的独立修正；身体接入 FASHN 18 类像素解析，结合 Pose 竞争分区得到真部位 mask，并将手、包、头发等作为固定层。分割置信度仍不等于物理 matting，宽袖和外套归属仍有失败。去人保留浏览器 ONNX LaMa 与 Go→Torch Worker；新测 PatchMatch/MI-GAN 未带来可靠净收益，不接入生产。逐功能 8 分目标尚未达到，见 [质量目标](qa/quality-targets.md) 与各项 QA。

版本：v0.2 · 2026-10-12

本期实现 Web 手工修图、Go 渲染 API 和可选 AI 工具入口。数据系统、商业化和原生客户端不进入本期。所有像素操作均由同一套 Rust 核心执行，前端参数面板、HTTP 和 MCP 使用同一份 Operation 协议。

2026-10-15按工具功能细化算法：像素蛋糕公开手册用于建立功能行为对照，不能作为其私有算法证据。整体宽度、整体尺寸、V脸、短脸及五官组合预设分开实现；对应[FUNC-024](func/FUNC-024-whole-face-shape.md)。全工具的定位、算法差距、验收和精细调用需求见[FUNC-025](func/FUNC-025-tool-algorithm-ledger.md)，优先算法实图效果，暂不扩展MCP工作流。

## 已落地的技术选择

| 环节 | 当前实现 | 原因与边界 |
|---|---|---|
| Web 界面 | React、TypeScript 严格模式、Vite | 工具面板、画布交互、操作图层、历史；无像素算法重复实现 |
| 图像核心 | Rust `image-core` + `image` + serde | 算法共用；`image` 负责 PNG/JPEG/WebP 解码编码、裁剪、旋转、Lanczos 缩放 |
| 浏览器执行 | wasm-bindgen + Dedicated Web Worker | RGBA 与 EditGraph 输入，RGBA 输出；计算离开 React 主线程 |
| 画布展示 | Canvas 2D + SVG 交互覆盖层 | Demo 先完成 CPU 参考路径；当前没有 WebGL2/wgpu 渲染器 |
| 服务端 | Go + Gin + Viper | HTTP、临时文件、revision、提案、任务状态与取消；不写 Go 像素循环 |
| 后端像素执行 | 同一 Rust crate 的 `image-render` native CLI | `exec.CommandContext` 调用，限定并发、超时、临时目录，失败不回传未编辑原图冒充成功 |
| 本地检测 | `@mediapipe/tasks-vision` | Face mesh、Pose 33点、人体分割 alpha 与部位 polygon；本地模型，可选，不等于衣物/皮肤解析 |
| 身体像素解析 | FASHN Human Parser（SegFormer-B4）+ ONNX Runtime Web | 18类概率、颜色感知测地线和骨轴竞争分区；独立串行Worker；不是现成左右袖/腿检测器 |
| MCP | 官方 `modelcontextprotocol/go-sdk` | `/mcp` Streamable HTTP；协议 SDK 管理初始化、会话和工具调用 |
| AI 编排 | Python、LangChain DeepAgents、langchain-mcp-adapters、FastAPI | 模型读取参数目录与预览图像并生成提案；UI 确认后由 Go 应用 |
| 开发与验收 | Makefile、Cargo、go test/vet/race、Playwright | 编译 WASM/native、实际文件上传、画笔、像素比较和错误恢复 |

Go 适合 API 和任务编排，Rust 适合浏览器 WASM 与 native 共用图像算法。Go WASM 不作为本期像素核心：运行时、包体和浏览器互操作不如 Rust/wasm-bindgen 方便，而且未来 GPU 接入仍需单独适配。

## 可采用、二次封装与不建议直接依赖

| 分类 | 项目 | 本期处理 |
|---|---|---|
| 可采用 | Gin、Viper、官方 MCP Go SDK、Rust `image`、serde、React、Vite、Playwright | 已使用成熟库，路由、配置、MCP 协议、编解码不重复实现 |
| 需要二次封装 | MediaPipe Tasks Vision | `landmarks.ts` 隔离模型加载、坐标归一化、Face 有序特征 polygon、Pose 部位支持域、人体 alpha 序列化/资源释放和失败重试 |
| 需要二次封装，已接入 | FASHN Human Parser、ONNX Runtime Web | 固定模型revision/hash，Torch→ONNX校验；bodyParsing/Worker处理输入、概率恢复、部位归属和独立保护；长外套与重叠同类衣物需手工校正 |
| 需要二次封装 | wasm-bindgen | `render.worker.ts` / `engine.ts` 隔离 WASM ABI，业务代码只传 Operation |
| 需要二次封装 | DeepAgents / MCP adapters | 仅暴露 inspect/preview/propose 图像工具；超时、无密钥和服务失败有明确响应；不把 AI 作为启动前置条件 |
| 需要二次封装，尚未接入 | `rust_mls`、`photon-rs` | 保留为替换算法候选；当前 Cargo 未依赖这两者，不能将自有反向采样称为它们的实现 |
| 不建议直接依赖 | Compositor 客户端骨架 | SwiftUI/AppKit/Metal 不能作为 Web 客户端代码；仅借鉴编辑图、图层和历史概念 |
| 不建议直接依赖 | Tract 作为现成美颜方案 | 它是推理运行时，仍需合适模型、算子支持和 Web 绑定；当前选择成熟ORT-Web执行公开ONNX模型 |
| 不建议直接依赖 | MediaPipe bbox 充当皮肤/衣物 mask | 区域框不具有像素语义；本期这些蒙版由用户手画 |
| 后续性能路径 | wgpu/WebGPU、WebGL2、Tile、代理图 | 当前尚未接入；已有 CPU 算法作为未来 GPU 对照，不声称实现 30 FPS 大图预览 |

完整库来源和使用边界见 [library-evaluation.md](library-evaluation.md)。本期按用户要求只验证开源 Demo，不进行商业化许可证审核。

## 当前架构

```text
React 工具 / 滑杆 / 画笔 / 图层 / 历史
  → EditGraph（原图引用 + 有序 Operation）
  → Web Worker → Rust WASM → RGBA → Canvas 2D

Go Gin / Viper
  → session / asset / graph / revision / proposal / job
  → Rust image-render CLI → PNG / JPEG

DeepAgent（可选）
  → Go /mcp inspect_asset / render_preview / propose_operations
  → EditProposal → 浏览器预览 → 接受 / 拒绝
  → 新 revision / 普通可撤销操作
```

源图始终不可变。重渲染从源图与操作序列开始；滑杆改变既有节点，不在上一次预览上继续加工。连续液化笔划作为操作内部序列回放，仍可能有重采样损失，后续需 WarpField 优化。

## 协议与参数

协议文件位于 `schemas/`，可执行参数目录从 Web registry 生成。修改类型或参数时必须同时更新 Rust 验证、Web UI、schema/catalog、Go 接口测试和 `doc/func/` 对账。

```json
{
  "version": "1.0",
  "sourceAssetId": "uploaded-asset-id",
  "operations": [{
    "id": "face-1",
    "type": "face.adjust",
    "version": "1.0",
    "enabled": true,
    "origin": "manual",
    "params": {
      "region": {"x": 0.2, "y": 0.1, "width": 0.6, "height": 0.8},
      "adjustments": {"faceWidth": 18, "jaw": 12},
      "opacity": 100,
      "blendMode": "normal"
    },
    "controlPoints": []
  }]
}
```

- 普通滑杆使用百分数。色阶 gamma 使用实际倍率，曲线点使用 0..1。
- region、mask、strokes、controlPoints 使用归一化坐标；crop/resize 使用像素。
- 画笔 radius 相对画布较短边；笔刷预览须按画布宽高换算。
- `mask` 限制生效范围，`protectMask` 冻结目标像素；支持矩形、椭圆、多边形、笔刷、线性和径向蒙版。
- 变更画布尺寸的几何操作不接受局部混合或局部蒙版，Rust 返回明确错误。
- 未知类型、参数拼写、非法范围和不合法数组会报错，不能静默忽略。
- 当前上限：32MP 图片、256 个操作、1024 个控制点、4096 个笔划点、65³ LUT。上限是资源保护，不是大图性能承诺。

## 算法能力

人脸在`face.rs`/`face_flow.rs`使用source→target控制、RBF/SVF和固定侧向QP；`faceContourMask/faceContour`是脸部控制边界，`face/faceOutline`是头部上下文。可信skin与外脸neck连通时默认共同传输上颈物理前景，低颈静止，缺项/断开保留原有效face-only/手工回退。QP source-v约束不能证明叠加chin/forehead竖向位移后的最终物理曲线普遍无凹凸或fold。BiSeNet概率经保留.5边界的coverage转换，分割仍不等于物理alpha；手工polygon是明确近似，裸矩形不猜人脸。

五官按实际semantic polygon在像素脸轴处理，避免非方照片的旋转剪切；鼻子矫正依据noseRoot→noseTip与脸轴的实际偏斜。身体粗细使用Pose部位轴和实际部位alpha，独立形变该部位RGB/alpha，背景与其他部位不位移；新露窄带按clean donor补背景。FASHN提供服装、肢体、头发、手、包等类别概率；颜色感知测地线与骨轴竞争将同类衣物划给torso/左右肢体，不能保证宽袖和外套遮挡归属正确。真部位mask存在时不再用旧近侧半径启发式剪窄不对称袖子。WASM前景/背景分解有单槽16MiB缓存，精确核对原图和alpha，图片/部位变动失效；native不缓存作为对照。没有分割、关键点或参数部位不匹配时拒绝执行。身高、腿长、体态和结构参数仍拒绝非零调用。

皮肤 mask 优先于 face，减去眼/眉/嘴/毛发与保护区域；磨皮使用有效皮肤donor的Fast Guided Filter明暗基底，并分开处理像素噪声、连贯毛孔/细纹、中尺度肤色与紧凑暗红异常。texture=100仍可修正部分异常；强磨皮在原图验收中仍有毛孔损失与蜡感，尚未证明全部纹理保留；痣、雀斑和痘印的可靠语义分类尚未解决。避免非皮肤或冻结颜色污染邻肤。美白与肤色分开调整亮度/色度，保护高光和深阴影。妆容只作用唇/眉/眼下等语义区域，同层先调色再几何采样以跟随特征变化；对称是可靠语义下的保守二维特征位置矫正。检测在当前层输入执行，旧上游几何/顺序变动后需重检测。牙齿在内嘴中独立筛选明亮低饱和像素。衣物平整使用低频与纹理混合。这些是确定性传统算法，尚非深度精修论文模型的完整复现。

LaMa 双端方式为浏览器 ORT-Web（固定512正方形输入，保留照片比例后对称补边，约208MB权重）与 Go 代理本地 Python/Torch Worker（默认 full/1024，支持 Poisson）。两端都由 Rust生成 canonical mask并合成一次；客户端没有静默远程回退。执行方式和参数进入同一Operation/MCP catalog。浏览器模型并不需要把图像算法改写成 Go；Go继续负责Gin/Viper、任务、缓存和协议，模型运行时用成熟ORT/Torch。PatchMatch与MI-GAN的实测失败保留在 [补全实验](qa/inpaint-refinement-review.md)，不因论文更新就替换现有输出。运行方式、限制、候选库及已验证论文见 [FUNC-016](func/FUNC-016-local-model-inpaint.md)、[研究记录](research/lama-client-runtime.md)。

2026-10-15 唇亮度采用已验证的线性RGB共同比例调节：正向色域肩部避免暗纹漂白，负向沿用中性高光保护；不依赖模型或新增运行库。五原照55组及等亮度消融支持该项改善，native/WASM/API五档一致并已更新本地服务；强负向蒙版边界仍未修复。见[FUNC-015](func/FUNC-015-skin-and-makeup-quality.md)与[接入验证](qa/lip-lighting-v2-integration-20261015/review.md)。整头ZIM覆盖与有界F/B分解只在隔离实验，未因数学合同通过就替换瘦脸生产路径。

修复笔刷可手动取样或自动选择邻近纹理；背景人物/杂物提供手工 polygon 圈选与主体冻结。内容填补使用confidence×isophote优先级和masked SSD匹配完整clean exemplar纹理块，保护和透明RGB排除在donor之外。狭窄背景自适应缩小匹配/复制窗；无合法patch明确报错。街景仍有块状接缝，不生成复杂缺失语义。功能与验收见 [FUNC-012](func/FUNC-012-body-semantic-masks.md)、[FUNC-014](func/FUNC-014-background-person-removal.md)、[FUNC-015](func/FUNC-015-skin-and-makeup-quality.md)。

胶片模拟包含自定义色彩风格、颗粒、暗角、辉光、halation、褪色和 3D `.cube` LUT。它是类似胶片模拟的风格组合，不是富士官方胶片算法或颜色复刻。

## 服务端和 AI

HTTP 接口与配置参数以 [services/api/README.md](../services/api/README.md) 为准。临时 session、提案、历史与 job 保存在内存；上传和渲染文件存临时目录，没有数据库或对象存储。

native Worker 使用 `--input --output --graph --format --quality`；stdout 输出结果元数据，像素文件通过路径交接。Go 使用请求 context，超时或取消终止进程，限制并发并清理临时目录。异步 job 提供阶段进度、结果下载和取消。

MCP 工具提供 inspect、propose、apply、render、export 和 undo。server 侧检测工具当前明确返回 unavailable，不返回假关键点。Web 本地模型有独立检测入口。

DeepAgent 服务启动不要求密钥；没有 `OPENAI_API_KEY` 时 `/recommend` 返回 503。配置 `IMG_AI_MODEL` 和可选 `OPENAI_BASE_URL` 后可调用兼容模型。真实付费模型推荐尚需配置后验收。Python 服务不接触手工渲染热路径。

## 工程保存与离线边界

Demo 导出 `.edit.json` 保存操作，不打包原图；恢复时先导入原图再载入工程。自定义滤镜预设用 localStorage。没有自动保存、崩溃恢复或云同步。

已加载 Web 页面、WASM 和静态资源后，断开 API/AI 可以继续导入、手工编辑、撤销和导出。首次离线打开页面仍需本地 Web 服务器；当前没有 PWA Service Worker，不能声称离线安装完成。

## 开发与验证

```bash
make setup       # JS依赖、WASM工具、Chromium与本地检测模型
make body-models # 可选：导出校验人体解析模型，约257MB
make dev         # 构建 native/WASM/API，启动本地 Web + Go
make test        # Rust、Go、vet、TS与生产构建
make e2e         # 浏览器实际交互与像素检查
make agent       # 可选 DeepAgent 服务
make stop        # 仅停止本工程启动并记录的进程
```

验证证据与当前完成状态见 [func/README.md](func/README.md) 和 [qa/phase1-validation.md](qa/phase1-validation.md)。

## 后续扩展

现有 EditGraph 与 MCP 不绑定浏览器 UI。桌面端可用 Tauri/Rust，移动端以 native Rust bindings 或平台 renderer 复用协议；云端可让 Go 调度容器化 native/GPU Worker。后续再引入数据库、S3、任务队列、GPU Tile、色彩管理和生成式 AI，不作为当前 Demo 依赖。
