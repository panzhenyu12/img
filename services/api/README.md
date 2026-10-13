# image-editor API

Go 使用 Gin、Viper 和官方 MCP Go SDK。图片算法、解码和编码统一交给 Rust `image-render` native Worker；Go 只做格式/尺寸元数据检查、协议、临时 session、文件管理、任务状态与工具调用。session、提案和任务保存在内存，重启后丢失。

## 启动

先从仓库根目录构建 Rust Worker，再启动 API：

```bash
cargo build --release --manifest-path core/image-core/Cargo.toml --bin image-render
cd services/api
go run ./cmd/server
```

配置读取工作目录 `config.yaml`，环境变量优先：

| 变量 | 默认值 |
|---|---|
| `IMG_API_ADDR` | `:8080` |
| `IMG_API_DATA_DIR` | 系统临时目录下的 `image-editor-api` |
| `IMG_API_SESSION_TTL` | `24h` |
| `IMG_API_MAX_UPLOAD_BYTES` | `104857600` |
| `IMG_RENDER_BINARY` | 自动查找仓库 `core/image-core/target/release/image-render` |
| `IMG_RENDER_TIMEOUT` | `2m`（包含排队时间） |
| `IMG_RENDER_CONCURRENCY` | `2` |

Worker 不存在时，只有空编辑图的同格式源图可以下载；含任何 Operation（包括 disabled 和 identity）或需要转码的请求返回 `503 render_worker_unavailable`，不会静默绕过校验。

## HTTP API

```text
POST   /api/v1/sessions
GET    /api/v1/sessions/{sessionID}
DELETE /api/v1/sessions/{sessionID}
POST   /api/v1/sessions/{sessionID}/assets  multipart field: file
GET    /api/v1/sessions/{sessionID}/source
POST   /api/v1/sessions/{sessionID}/preview
POST   /api/v1/sessions/{sessionID}/render
PUT    /api/v1/sessions/{sessionID}/graph
POST   /api/v1/sessions/{sessionID}/operations
POST   /api/v1/sessions/{sessionID}/undo
GET    /api/v1/sessions/{sessionID}/proposals
POST   /api/v1/sessions/{sessionID}/proposals
POST   /api/v1/sessions/{sessionID}/proposals/{proposalID}/apply
POST   /api/v1/sessions/{sessionID}/proposals/{proposalID}/reject
POST   /api/v1/jobs
GET    /api/v1/jobs/{jobID}
GET    /api/v1/jobs/{jobID}/result
POST   /api/v1/jobs/{jobID}/cancel
GET    /api/v1/tools
GET    /api/v1/operations
POST   /api/v1/tools/{toolName}/call
GET    /healthz
```

`preview/render` JSON：

```json
{"editGraph":{"version":"1.0","sourceAssetId":"uploaded-asset-id","operations":[{"id":"color-1","type":"color.basic","version":"1.0","enabled":true,"params":{"exposure":20}}]},"format":"png","quality":92}
```

源 asset 必须匹配 session；为旧 demo 保留省略 `sourceAssetId/id/version/enabled` 的兼容。`format` 只允许 `png/jpeg`（`jpg` 别名）；Rust 未实现的参数、非法参数或 operation 返回 `422`。请求取消会杀掉 Worker，超时返回 `504`；每请求的 graph/input/output 临时目录在完成和失败后都清理。

源字节直通只适用于没有任何 Operation 的同格式图。含节点均先执行 Rust 校验/渲染，disabled 和 identity 的坏键或非法值也返回 422。仅当全部节点 disabled 且同格式时，Rust 校验成功后返回源字节，保留关闭效果的原编码。enabled identity 始终使用 Rust 输出，因为其 blendMode 等通用合成参数可能产生实际效果。

上传返回的 `width/height` 与 `inspect_asset.asset` 是展示方向尺寸。JPEG 使用 `goexif` 读取 EXIF Orientation，5/6/7/8 交换宽高，与浏览器和 Rust 的方向校正一致；Go 不旋转像素，源 JPEG 字节保留。可选 EXIF 不存在或损坏时使用 JPEG 原尺寸。

`POST /jobs` 使用同一 JSON，额外必填 `sessionId`。也可以用 `/sessions/{id}/jobs`。返回 `202 {id,status,progress,stage}`，状态是 `queued/running/succeeded/failed/cancelled`。进度是阶段标记（排队、运行、编码、完成），不表示每像素百分比。完成后从 `resultUrl` 下载。

## 编辑图与提案

`GET session` 返回 `{id,image,revision,editGraph}`。Web 滑杆先本地预览，发起 AI 推荐前同步：

```json
{"baseRevision":"1","editGraph":{"version":"1.0","sourceAssetId":"asset-id","operations":[]}}
```

向 `PUT /sessions/{id}/graph` 发送，Rust 验证/渲染成功后返回新 `{revision,editGraph}`。人工 `POST /operations` 使用 `{baseRevision,operations}` 追加到同一编辑图。

提案接口 `POST /proposals` 接收 `{baseRevision,operations,summary}`；先由真实 Rust 核心验证与当前图合并后的参数，再返回 `201 EditProposal`。未知参数和非法值不会存成待确认提案。`GET /proposals` 返回 `{proposals:[...]}`。应用必须显式发送：

```json
{"accepted":true,"baseRevision":"current-revision"}
```

到 `POST /proposals/{id}/apply`。提案 base revision 必须匹配；Rust 先验证完整编辑图，成功后原子追加操作、保存上一图用于 undo，并返回新 `{revision,editGraph}`。过期、已应用和已拒绝提案不能应用。原图从不覆盖。

拒绝使用 `POST /proposals/{id}/reject` 和 `{baseRevision}`，返回 `{proposal,revision,editGraph}`。拒绝将状态从 `proposed` 改为 `rejected`，不修改图、不新增 revision。即使应用正在等待 Rust 校验，拒绝也会阻止它之后提交。`apply` 省略 accepted 或传 false 始终返回错误。

## MCP

`/mcp` 使用 `github.com/modelcontextprotocol/go-sdk` 的 Streamable HTTP，支持 MCP initialize、tools/list、tools/call、会话和 JSON/SSE。DeepAgent/MCP Client 直接连接 `http://localhost:8080/mcp`。

已注册：`inspect_asset`、`propose_operations`、`apply_operations`、`render_preview`、`export_image`、`undo_operations`。`detect_landmarks` 和 `detect_regions` 返回 `browser_provider_required`，由浏览器 MediaPipe 提供，不返回伪造结果。

所有工具参数都包含 `sessionId`。`inspect_asset` 返回当前 `revision/editGraph/asset` 及生成的 operation 参数目录；`propose_operations` 接收 `baseRevision/operations/summary`，返回 `pending_approval + proposal`。`apply_operations` 需要 `proposalId/baseRevision/accepted:true`。`undo_operations` 需要 `baseRevision/accepted:true`。`render_preview` 通过 MCP ImageContent 返回真实预览图，`export_image` 返回可查询、可取消的 job。

工具结果沿用 `internal/ai.ToolResult`；工具错误使用 `isError` 与结构化错误，不返回服务器文件路径或 Worker 堆栈。当前 Demo 未实现账号鉴权与云持久化。

## 验证

```bash
GIN_MODE=release go test ./...
GIN_MODE=release go test -race ./...
go vet ./...
```

测试包含同一真实人像上全部 22 类 Operation 的 HTTP/Rust 渲染、局部修复/变形/画笔的保护蒙版逐像素核对、官方 SDK 的完整 HTTP MCP 调用链、源 asset 不匹配、提案接受/拒绝和 revision 冲突、拒绝与应用竞态、提案原生参数校验、任务下载/取消、Worker 超时及上传删除竞态。实际 Rust 渲染测试前需要先构建 Worker。
