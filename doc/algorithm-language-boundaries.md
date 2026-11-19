# 当前算法语言边界

核对日期：2026-11-19。本文描述当前源码和实际调用入口；语言实现、效果验收、Web 接入是三个不同状态。

| 功能或环节 | 当前语言与运行方式 | 边界 |
|---|---|---|
| 裁剪、旋转、缩放、透视、网格与液化 | Rust `image-core` | 普通操作图由同一核心执行 |
| 瘦脸、脸型轮廓、位移场、采样与合成 | Rust | 新显式源材料入口仍依赖外部准备的源材料；不意味着整个自动流程已纯 Rust |
| 收腰、腿/手臂宽度调整、前后层合成 | Rust | 新身体三层算法由 Rust 运动和合成，源层准备仍用 Python |
| 五官几何调整 | Rust | 存在算法代码不等于所有项目已通过效果验收 |
| 新眼睑轻闭合 | Rust：源材料估计、场、采样与合成 | 普通 `prepare_eye_closure` 已不依赖 Python；仍需关键点及可选人工分类纠正，轻档有限验证；通用眼高/眼睑滑杆未开放 |
| 磨皮、美白、肤色、唇色、眉色、牙齿提亮 | Rust | 已有基础颜色与局部处理；不能据此声称完整腮红、眼影、修容已实现 |
| 衣物局部调整、平滑与颜色 | Rust | 有实现，不代表专业衣物去褶皱已经验收通过 |
| 曲线、HSL、胶片滤镜、LUT、画笔 | Rust | 普通操作图入口 |
| 仿制、修复、传统样本填补 | Rust | 与神经模型补全分开 |
| 服务端 LaMa 补全/去人 | Python 调用 PyTorch/TorchScript | 包括预后处理及可选接缝融合；最终请求范围和保护仍由核心控制 |
| 客户端 LaMa 补全/去人 | TypeScript 调用 ONNX Runtime Web | WebGPU/WASM 推理；Rust 负责编辑蒙版与最终保护合成 |
| 浏览器人脸/身体关键点、语义分割和部位派生 | TypeScript 调用 MediaPipe / ONNX Runtime，并进行几何与语义后处理 | 包括身体部位归属逻辑，并非只有 UI |
| 新脸/身体源材料准备 | Python 为主 | MEMatte 等模型推理、前后层/场景补全、透明度与颜色配对、材料角色审计；新身体人工方案仍依赖此阶段 |
| 研究候选、照片对比、边界量测、验收 | Python 为主，另有 Rust/TS 测试 | 历史实验文件不等于当前产品实际采用的算法 |
| API、MCP、任务与调用编排 | Go | 调用 Rust 原生核心及 Python 模型服务，不实现主要像素变形循环 |
| DeepAgents / LangChain 的 AI 修图建议 | Python | 生成或调用操作建议，不执行主要像素算法；需相应 API 配置 |

## 当前调用链

普通 Web 编辑：浏览器检测/参数准备 → Web Worker → Rust WASM 操作图 → 画布。

普通服务端编辑：Go API/MCP → Rust `image-render`；模型补全操作由 Go 编排调用 Python LaMa 服务。

本轮身体分层实验：人工源角色 → Python 源层准备 → Rust 原生 `adjust_body_with_layered_source` → 成片与独立验收。源材料准备并不是每次修改强度都必须重复执行的形变阶段。

`wasm.rs` 目前导出操作图和模型补全蒙版/合成接口；新的显式脸源缓存、身体三层和眼睑轻闭合入口尚未通过该文件直接暴露给 Web。不能把“Rust 已实现”写成“浏览器已更新”。最近工作仅针对算法，没有整体 Web/Go/WASM 构建或部署。

ONNX Runtime 的 WASM 推理后端与本项目 Rust 编译的 WASM 核心是不同运行库；使用 WASM 不代表该算法由本项目 Rust 实现。

当前 Go MCP 中的人像检测工具仍标为浏览器 provider 且服务端未启用。不能把已有 API/MCP 和客户端检测，表述为通用人像检测已经两端完成。

## 源码依据

- [普通 Rust 操作分发](../core/image-core/src/engine.rs)、[新源材料公共接口](../core/image-core/src/lib.rs)、[WASM 导出](../core/image-core/src/wasm.rs)。
- [Python LaMa 服务](../services/background-worker/worker.py)、[本轮身体源准备](../experiments/body-fresh-source-20261119/prepare_manual_layers.py)。
- [浏览器渲染](../apps/web/src/core/render.worker.ts)、[客户端模型补全](../apps/web/src/core/clientInpaint.ts)、[身体语义后处理](../apps/web/src/core/bodyParsing.ts)。
- [Go 渲染编排](../services/api/internal/api/render.go)、[MCP 工具](../services/api/internal/api/tools.go)。
