# 人像修图 Web Demo

开发中的手工修图 Demo：Web 前端 + Go API + Rust WASM/native 共用图像核心。AI 是可选入口；API 或 AI 不可用时，本地手工编辑、撤销和导出仍可使用。

## 本地运行

```bash
make setup     # 首次安装 JS、WASM 工具、Chromium 和本地识别模型
make dev       # 构建并启动 Web + Go
make health
make test      # Rust、Go test/vet、TypeScript 和生产构建
make e2e       # 浏览器实际操作、像素验证与双端一致性
make quality   # 多张真实照片对比；保留 FAIL / UNVERIFIED
make agent     # 可选 DeepAgents 服务，前台运行
make stop      # 停止本工程记录的 Web/API 进程
```

Web：<http://127.0.0.1:5173>；API：<http://127.0.0.1:8080>；MCP：<http://127.0.0.1:8080/mcp>；可选 Agent：<http://127.0.0.1:8090>。代码更新后的 API 重启使用 `make stop && make dev`。需要 Node 24+、Go、Rust/rustup、Python 3；AI 服务另需 uv。自动检测模型由 `make models` 放到本地静态目录。

## 已实现的 Demo 功能

- 导入 JPG/PNG/WebP，缩放、平移、原图与分屏对比；PNG/JPEG 导出及 JPEG 质量。
- 已接入的滑杆以自动生成的 [参数对账](doc/func/FUNC-010-parameter-ledger.md) 为准；未验证的结构参数已隐藏，并在核心拒绝非零调用。
- 本地 MediaPipe Face/Pose 定位、BiSeNet ONNX 脸部/皮肤/头发解析、人体分割；手工矩形、椭圆、套索、多边形和冻结区。
- 液化推移/收缩/膨胀/双向旋转/重建/对称；3×3 网格、四角透视、裁剪、旋转、翻转、尺寸调整。
- 仿制、取样选区修补、邻近纹理修复、传统内容填补、绘色/橡皮；蒙版涂抹/擦除/反选/羽化和渐变。
- 曲线、分区 HSL、8 种胶片风格及真实缩略图、自定义预设、3D `.cube` LUT。
- 有序非破坏操作层、混合与透明度、开关/删除/重排/重置、undo/redo、参数复制与工程 JSON 保存/载入。
- Go Gin/Viper 临时会话与上传、native 同步/异步渲染、任务取消、revision 与提案；官方 SDK Streamable HTTP MCP。
- 可选 LangChain DeepAgents 查看图片与目录并生成提案，Web 预览/接受/拒绝；没有密钥时明确提示且不影响手工链路。

浏览器不实现另一套 JS 像素算法；22 类 Operation 由同一 Rust crate 执行，服务端 Go 负责 API 和任务编排。

## 目录与文档

```text
apps/web/             React + TypeScript + Vite / Playwright
services/api/         Go Gin + Viper + 官方 MCP SDK
services/agent/       可选 Python DeepAgents + FastAPI
core/image-core/      Rust WASM / image-render native CLI
schemas/              EditGraph / Operation / mask / 参数目录
scripts/              启停、模型安装、目录生成、多照片诊断
test/images/          真实测试照片、来源与 SHA256
test/results/         可重生成对比图与操作图（不入 Git）
doc/                 产品、技术、一期计划
doc/func/            独立功能实现与对账
doc/qa/              验收记录与截图
```

[一期计划](doc/phase1-development-plan.md) · [技术选型](doc/technical-selection-development.md) · [库评估](doc/library-evaluation.md) · [功能对账](doc/func/README.md) · [验收记录](doc/qa/phase1-validation.md) · [MCP/AI](doc/mcp-ai-integration.md)。完整未来功能目录见 [feature-list](doc/feature-list.md)，其优先级不是本期完成标记。

## 当前边界

目前为 CPU WASM/native + Canvas 2D，没有 wgpu/WebGPU、瓦片或 30FPS 大图性能验收。脸部与身体轮廓使用前景 alpha 分离及背景窄带重建，五官基于实际语义部位和像素坐标。单测验证方向、轮廓/面积、背景、冻结和透明度；真实照片效果仍未通过全部验收，部分样片有色缝、遮挡定位或背景取样缺口。见 [算法验证与缺口](doc/qa/algorithm-validation.md) 与 [当前测试/性能记录](doc/qa/current-validation.md)。

去人使用 exemplar 纹理块补全，街景可移除选中人物，复杂纹理仍有接缝，尚无生成式语义重建。衣物功能是褶皱细纹弱化，不能重建严重褶皱或版型。不能以“像素变化”“单测通过”声称自然度达到专业或世界领先水平。

工程 JSON 只保存操作，恢复需提供原图。无账号、数据库、云同步、RAW/16-bit/ICC 和完整 PSD 图层。8 种胶片风格用于效果模拟，未声称精确复刻富士官方配方。已加载页面可独立手工编辑；首次离线打开仍需本地服务器，未实现 PWA。
