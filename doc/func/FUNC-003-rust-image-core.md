# FUNC-003 Rust 图像核心与 WASM 边界

状态：已验证（shared engine，2026-10-12）

关联需求：CORE-007、LAYER-009、WARP-014；完整能力对账见 [FUNC-007](FUNC-007-shared-rust-render-engine.md)。

## 用户行为与前后端实现

用户提交同一 EditGraph 到浏览器预览或 Go API 导出。Rust `image-core` 的 `apply_graph` 是两条路径唯一的像素执行器：Web Worker 调 wasm-bindgen `render_rgba`，Go 用受限 native 子进程调用 `image-render`。UI 与 Go 都不复制像素算法。

实现：`core/image-core/src/engine.rs`、geometry/color/mask/paint/validation、wasm、bin/image-render；前端 `engine.ts` / `render.worker.ts`。

## Operation 与参数

当前22种类型见生成的 `schemas/operation-catalog.json`。滑杆通常为百分数，region/mask/strokes/controlPoints归一化0..1，crop/resize为像素。未知参数、非法数值、非法数组甚至禁用节点内的错误都会被校验，避免隐藏非法操作以后启用失败。

## 完成标准与测试

- native/WASM 均真实构建，不再是预留 cdylib。
- 同一 PNG 与含人像、调色、颗粒、裁剪的同一工程，浏览器与 native 逐像素一致。
- 原图不可变、alpha、蒙版冻结、paint橡皮、空操作与非法参数有 Rust 测试。
- 31个 Rust 测试、Clippy 与浏览器加载/失败恢复已验证。

## 当前限制与变更

当前是 CPU 参考路径；无 wgpu/WebGL2/Tile/OffscreenCanvas。浏览器支持预览合并和 Worker 终止取消；导出独立排队，不被滑杆取消。2026-10-12 升级至 image-core 0.2.0 与 shared JSON engine；旧 `lib.rs` 的基础 Rust 类型仅兼容早期测试，不是 Web/native 参数协议。
