# FUNC-001 编辑器基础链路

状态：已验证（Web 手工 Demo，2026-10-12）

关联需求：CORE-001（部分格式）、CORE-007、CORE-008、CORE-009、CORE-010、CORE-012、CORE-013、CORE-014、CORE-015；完整操作 UI 见 [FUNC-008](FUNC-008-web-editor-tools.md)。

## 用户行为

导入 JPG/PNG/WebP，选择工具，拉滑杆或在画布拖动；查看操作层、开关、重置、撤销重做；保存操作工程并导出 PNG/JPEG。加载后的页面不依赖 API 或 AI 完成这些动作。

## 实现与协议

前端：`apps/web/src/App.tsx`、CanvasStage、ToolPanel、Sidebar、useHistory。Canvas 2D 展示原尺寸像素，SVG 处理区域、笔刷与控制点；参数作为 EditGraph 记录，Rust WASM Worker 执行像素。

后端：可选创建临时 session/上传；前端导出直接使用 WASM。API/native 独立提供同一工程渲染，当前 Web 导出按钮不自动走后端。

Operation：由 `core/operations.ts`、`registry.ts`、`schemas/operation.schema.json` 定义。原图输入不可变，操作顺序决定结果。连续滑杆记录合并，普通动作可逐步撤销。

## 完成标准与测试

- 图片显示、缩放/平移、适应窗口与真实100%可操作。
- 调色真正改变像素；撤销、重做、关闭节点恢复对应结果。
- 导出尺寸来自图像结果，视图缩放不改变导出尺寸。
- 保存 `.edit.json` 后可在同一原图上恢复；非法工程校验失败不覆盖当前编辑。
- Playwright 已覆盖断开 API 的导入、真实像素调整、历史、工程和导出；验证记录见 [QA](../qa/phase1-validation.md)。

## 当前限制与变更

工程 JSON 不包含原图；无自动保存/PWA/完整图库。当前 CPU WASM，不把 Canvas 展示当作 GPU 渲染。2026-10-12 从基础调色切片扩展到完整手工 Demo，撤回旧文档中“JS Canvas fallback 算法”和“导出优先后端”的描述。
