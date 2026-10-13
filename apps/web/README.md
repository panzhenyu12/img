# 人像修图 Web 编辑器

React + TypeScript + Vite，所有图像效果通过 Worker 调用 Rust WASM；Canvas 2D 展示结果，SVG 接收区域、蒙版、笔刷与控制点手势。

从仓库根目录运行 `make setup`、`make dev`；打开 <http://127.0.0.1:5173>。仅启动前端可使用 `make web`，它会先生成 WASM。Go 不可用时进入本地修图模式，不影响手工编辑与 PNG/JPEG 导出。

支持脸型/五官/皮肤/身形/衣物滑杆、液化/网格/透视、修复/仿制/绘色、蒙版/冻结、调色/曲线/HSL/胶片/LUT、非破坏历史和工程 JSON。具体范围和真实约束见 [FUNC-008](../../doc/func/FUNC-008-web-editor-tools.md) 与 [当前算法验证](../../doc/qa/algorithm-validation.md)。

```bash
npm run build
npx playwright test  # 需先从根目录 make dev
```

`core/engine.ts` 管理最新预览合并与 Worker 取消、导出快照及后台缩略图队列。`core/registry.ts` 是滑杆和 AI 参数目录来源；修改后在根目录运行 `make catalog`。`make models` 安装本地 MediaPipe与BiSeNet ONNX模型，检测失败可继续手工定位。

DeepAgent 服务可单独通过 `make agent` 启动，未配置模型时返回503。AI 预览为临时图，接受前检查本地与服务端图版本，拒绝不修改历史。浏览器自动化测试不调用付费模型。
