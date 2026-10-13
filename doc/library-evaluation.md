# 开源图形库采用与封装清单

版本：v0.2 · 2026-10-12

本文区分“可采用”与“仓库已经采用”。用户授权先做开源 Demo，本期不做商业化许可证审查。

## 可采用项

| 项目 | 本期状态 | 用途 |
|---|---|---|
| [Gin](https://github.com/gin-gonic/gin) + [Viper](https://github.com/spf13/viper) | 已采用 | Go 路由、中间件、配置和环境变量 |
| [MCP Go SDK](https://github.com/modelcontextprotocol/go-sdk) | 已采用 | Streamable HTTP MCP 协议，不手写 JSON-RPC transport |
| [Rust image](https://github.com/image-rs/image) + serde | 已采用 | PNG/JPEG/WebP、像素模型、裁剪旋转、Lanczos resize与 JSON |
| [MediaPipe Tasks Vision](https://github.com/google-ai-edge/mediapipe) | 已采用 Face/Pose | 可选本地脸/身体关键点与区域框 |
| wasm-bindgen、React、Vite、Playwright | 已采用 | WASM ABI、Web UI、构建和实际浏览器验收 |

## 需要二次封装项

| 项目 | 封装要求 | 代码与限制 |
|---|---|---|
| MediaPipe | 模型加载、缓存、坐标归一化、失效重试；Face mesh 与 Pose 语义部位 polygon | `apps/web/src/core/landmarks.ts`；只接 Face/Pose，不假装具有衣物分割 |
| wasm-bindgen | Worker消息与 Rust ABI 适配 | `engine.ts`、`render.worker.ts`；前端不复制算法 |
| DeepAgents / langchain-mcp-adapters | 限定工具、提案、超时、失败回退 | `services/agent/main.py`；仅开放图像 inspect/propose，应用仍由用户确认 |
| [rust_mls](https://github.com/mpizenberg/rust_mls) | 点格式、反向采样、蒙版、native/WASM、版本和资源上限 | 候选，尚未依赖；现有 grid 是加权位移，不称为 MLS |
| [photon-rs](https://github.com/silvia-odwyer/photon) | 参数单位、蒙版合成、透明度、随机seed与双端一致性 | 候选，尚未依赖；当前确定性滤镜由 shared image-core 实现 |
| wgpu/WebGL2 | 同一Operation下的 CPU/GPU 对照、纹理与Tile管理 | 后续加速，当前 Demo 只有 CPU WASM/native |

“库里有函数”不代表可以直接成为可撤销功能。所有库需要被统一 Operation 包裹，控制点、蒙版、参数范围与随机种子必须可保存、回放。

## 不建议直接依赖项

- [Compositor](https://github.com/robbietilton/compositor)：原生 macOS 界面和图像框架不能直接当 Web 骨架。可参考非破坏编辑、图层、蒙版和历史设计，未来 Mac 实验再评估。
- [Tract](https://github.com/sonos/tract) 作为现成美颜：它是模型推理框架，还需要适用模型、算子覆盖、WASM绑定和效果验收。本期没有自有 ONNX 模型，不为运行时引入额外依赖。
- 将关键点检测当作瘦脸算法：MediaPipe 输出坐标；瘦脸需要独立几何规则、采样和保护。
- 将人脸/身体 bbox 当作皮肤/衣物 mask：bbox 只帮助用户选区域，不能做语义像素选择。
- 让 AI 或外部 CDN 成为手工功能的必选依赖：模型失败只能关闭自动入口。检测模型和runtime由 `make models` 放到本地静态目录。

## 当前实际链路

```text
用户手动选区 / 可选 MediaPipe 区域框
  → Operation 中的 region / adjustments / controlPoints / mask
  → 自有 Rust 反向采样、局部平滑、颜色与修复算子
  → 同一crate编译 WASM 与 native
  → Canvas 2D / PNG-JPEG
```

当前人脸轮廓真正实现 Schaefer rigid MLS 公式及 Germer 多尺度 F/B 估计，未依赖未经核对的 rust_mls 包；Web采用BiSeNet ONNX类别解析，但分类覆盖不能当真实matting。身体采用Pose轴/真实subjectMask及独立目标partMask接口，FASHN/SCHP仍需部位封装和质量验收，低指标B0已否决。皮肤采用有蒙版Fast Guided Filter；衣物仍仅细褶皱弱化。LaMa服务端TorchScript已核原权重一致性；Carve固定512 ONNX可采用为浏览器推理模型，需二次封装预处理、保护、合成、缓存与错误恢复。直接torch.onnx.export实测不支持aten::fft_rfftn，不能把原Torch模型文件交给浏览器直接运行；Carve用矩阵DFT替换FFT，已做同输入数值核对。MAT/BrushNet/PowerPaint保持候选，未运行不得标记已采用。GPU renderer和学习式皮肤/衣物精修尚未完整交付。具体实现对账见 [func/README.md](func/README.md)。
