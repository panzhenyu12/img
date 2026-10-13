# FUNC-002 图片导入与临时渲染

状态：已验证（2026-10-12）

关联需求：CORE-001（JPG/PNG/WebP 子集）、CORE-003（尺寸/EXIF方向）、CORE-007；导出对应 LAYER-010（PNG/JPEG 子集）、LAYER-011（尺寸/质量）、LAYER-012。

## 用户行为与实现

前端先本地解码显示，再异步创建 Go session 上传；网络失败保留本地编辑，上传完成仅更新 sourceAssetId，不能覆盖等待期间的手工操作。后端使用 Gin multipart、文件大小与格式检查、Viper 配置。Go 读取尺寸与方向元数据；真实解码、像素算法与 PNG/JPEG 编码全部交给 Rust。

实现：`apps/web/src/core/engine.ts`、`api/client.ts`；`services/api/internal/api/server.go`、`render.go`、`jobs.go`；native CLI。

## 接口与参数

```text
POST /api/v1/sessions
POST /api/v1/sessions/{id}/assets      multipart file
GET  /api/v1/sessions/{id}/source
POST /api/v1/sessions/{id}/preview
POST /api/v1/sessions/{id}/render
DELETE /api/v1/sessions/{id}
```

render 使用 `{editGraph:{version,sourceAssetId,operations},format:'png'|'jpeg',quality:92}`。sourceAssetId 必须对应 session，所有当前22类 Operation 使用同一 Rust 执行器。任务/取消/MCP 见 [FUNC-009](FUNC-009-go-render-mcp.md)。

## 完成标准与测试

Go 测试覆盖上传、无效图像、超限、过期 session、删除竞态、asset绑定、全部类型真实 native 渲染与错误返回；Playwright 覆盖本地导入、PNG/JPEG 下载和 WASM/native 逐像素比较。EXIF 方向读取只影响展示元数据，原文件不改写。

## 当前限制与变更

无数据库和云存储；内存 session/提案/job 重启丢失。Web 只提供 JPG/PNG/WebP 入口，RAW/TIFF/HEIF 不属于本期。2026-10-12 将旧 Go Demo 像素实现替换为 Rust native；新增 job、取消与 EXIF 方向适配。

## 2026-10-13 本地与服务端处理更新

Web普通导入只解码到本地RGBA与EditGraph，不再自动创建session或上传图片。用户在AI助手点击“连接 AI 服务（上传图片）”时才创建session并上传原图；客户端模型填补由浏览器完成，服务端模型按钮明确发送该操作输入与蒙版。普通编辑、工程保存和图片导出保持本地可用。验证：client-inpaint.spec.ts普通导入API POST数量为0、editor AI提案测试通过显式连接继续走原提案流程。
