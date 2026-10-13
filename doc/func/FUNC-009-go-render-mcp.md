# FUNC-009 Go 原生渲染、任务与 MCP 网关

状态：已验证（Demo）

关联需求：

- `CORE-007` 原图不可变
- `CORE-014` 撤销、重做和历史
- `CORE-015` 手工参数与 AI 共享参数编辑
- `LAYER-009` 非破坏编辑与参数重算
- `LAYER-010` PNG / JPEG 导出
- `LAYER-012` 原尺寸导出
- `LAYER-015` 导出任务状态和取消（Demo 内存任务）
- `LAYER-016` AI 结果使用独立 Operation
- `FACE-027` AI 推荐面部参数（工具桥接，不含模型推理）

实现位置：

- [Go 服务](/Users/modao/work/img/services/api/internal/api/server.go)
- [Rust Worker 调用](/Users/modao/work/img/services/api/internal/api/render.go)
- [任务管理](/Users/modao/work/img/services/api/internal/api/jobs.go)
- [MCP 工具与编辑事务](/Users/modao/work/img/services/api/internal/api/tools.go)
- [AI 共享协议](/Users/modao/work/img/services/api/internal/ai/protocol.go)
- [服务端接口说明](/Users/modao/work/img/services/api/README.md)

## 用户行为

用户导入图片后，Web 在本地完成实时编辑。需要后端导出时向 Go 提交同一份 EditGraph；需要 AI 推荐时先保存本地编辑图，再由 DeepAgent 通过 MCP 查看当前版本和参数目录、生成提案。用户确认后追加提案，仍可撤销。导出可以同步等待，也可以创建任务并查询或取消。

## 前端实现

前端 API 封装提供 `/sessions/{id}/render` 调用，浏览器集成测试用它验证后端 PNG 输出；当前导出按钮默认使用本地 WASM。后端 JPEG、异步任务和取消由 HTTP/MCP 接口提供，尚未做独立任务管理页面。为让 AI 查看当前人工编辑上下文，推荐前调用 `PUT /sessions/{id}/graph`，传入 `baseRevision` 和 `editGraph`。获取提案后显式确认 `accepted:true`，接收新 revision 和完整编辑图。拒绝按钮调用独立的 `/proposals/{id}/reject`，保存本次内存会话的提案状态。

前端持续拖动滑杆时使用本地 Rust/WASM，不需要每帧调用 Go。AI 或网络关闭时本地手工链路继续可用。

## 后端实现

Gin 负责路由，Viper 负责 YAML/环境变量配置；MCP 使用官方 `github.com/modelcontextprotocol/go-sdk v1.8.0`。Go 不实现磨皮、调色、变形或像素编码循环。

Rust Worker 契约：

```text
image-render --input PATH --output PATH --graph PATH --format png|jpeg --quality 92
```

Worker 与浏览器 Rust/WASM 使用同一图像核心。stdout 只读取宽、高、格式元数据；错误非零退出转为 `422`，先去掉路径再返回用户。每请求独立创建 input 快照、graph JSON 和 output 文件，完成、失败和取消后均清理。

`IMG_RENDER_BINARY` 指定 Worker；省略时自动查找仓库 release binary。默认最多同时运行 2 个 Worker、超时 2 分钟。使用 `exec.CommandContext` 取消进程；正在排队的渲染也响应取消和超时。

临时 session 的读取在锁内复制，上传提交前再次确认 session 存在。替换上传时保留旧源文件至 session 清理，以免使正在执行的图像快照失效；删除与上传并发不会解引用不存在的 session。

上传使用 `github.com/rwcarlsen/goexif/exif` 读取 JPEG Orientation 元数据，5/6/7/8 交换展示宽高；上传和 AI `inspect_asset` 得到的尺寸与浏览器显示、Rust native 方向校正保持一致。Go 只读元数据，不旋转或重新编码像素，源文件原字节保持不变。可选 EXIF 不存在或损坏时回退 JPEG 原尺寸。

## Operation 类型

Go `Operation` 直接复用 `internal/ai.Operation`，人工、AI、HTTP 与 MCP 没有独立像素协议。支持 schema 的所有类型，并加入 `resize`、`color.hsl`、`filter.lut`、`repair.inpaint`、`paint.brush`。具体参数和执行以 Rust 核心校验为准。

`sourceAssetId` 必须匹配上传的 session asset。兼容早期请求省略此字段及 Operation `id/version/enabled`；其中省略 enabled 视为 true，明确 false 始终关闭该操作。

未知 operation、未知参数或非法参数返回错误，不静默输出原图。仅没有任何 Operation 的同格式空图可直接复制源字节。含节点均经过 Rust 校验/渲染；只有全部 disabled 且同格式时，校验成功后才返回源字节以保留关闭效果的原编码。enabled identity 也使用 Rust 输出，通用 blendMode 等合成参数可能真实改变像素，不能将它无条件视为零效果。

## 参数结构和接口

同步渲染：

```json
{
  "editGraph": {
    "version": "1.0",
    "sourceAssetId": "uploaded-asset-id",
    "operations": [
      {"id":"color-1","type":"color.basic","version":"1.0","enabled":true,"params":{"exposure":20}}
    ]
  },
  "format": "png",
  "quality": 92
}
```

异步 `POST /api/v1/jobs` 在同一请求中加入 `sessionId`。响应 `202 {id,status,stage,progress}`；`GET /jobs/{id}` 返回状态，成功后 `GET resultUrl` 下载，`POST /jobs/{id}/cancel` 取消。阶段进度为 `queued/running/encoding/done`，不宣称逐像素实时进度。内存队列上限 64 个进行中任务，状态/结果缓存上限约 256 项。

编辑事务：

| 方法 | 接口 | 请求 | 响应 |
|---|---|---|---|
| GET | `/sessions/{id}` | — | `{id,image,revision,editGraph}` |
| PUT | `/sessions/{id}/graph` | `{baseRevision,editGraph}` | `{revision,editGraph}` |
| POST | `/sessions/{id}/operations` | `{baseRevision,operations}` | `{revision,editGraph}` |
| POST | `/sessions/{id}/proposals` | `{baseRevision,operations,summary}` | EditProposal |
| GET | `/sessions/{id}/proposals` | — | `{proposals:[...]}` |
| POST | `/sessions/{id}/proposals/{proposalID}/apply` | `{baseRevision,accepted:true}` | `{revision,editGraph}` |
| POST | `/sessions/{id}/proposals/{proposalID}/reject` | `{baseRevision}` | `{proposal,revision,editGraph}` |
| POST | `/sessions/{id}/undo` | `{baseRevision}` | `{revision,editGraph}` |

路径统一加 `/api/v1` 前缀，revision 为字符串。提案创建、apply 和人工 graph 保存都会先调用 Rust 校验/渲染完整图，再复查 revision 并原子提交，避免验证期间的并发修改。提案创建也强制校验关闭的 operation，避免保存不可启用的参数。

拒绝只将 `proposed` 改为 `rejected`，编辑图和 revision 不变。apply 只允许 proposed 状态；Rust 返回后会再次核对提案状态，避免用户在渲染期间拒绝后仍被应用。已应用和已拒绝的提案均不可重复应用。省略 accepted 或传 false 不是拒绝接口，仍返回明确错误。

## MCP 能力

`/mcp` 为真实 Streamable HTTP MCP Server，支持 initialize、tools/list 和 tools/call。注册 8 个工具：

- `inspect_asset`：读取当前图、revision、asset 与生成的 Operation 参数目录。
- `propose_operations`：保存标准编辑提案，返回 `pending_approval`。
- `apply_operations`：只接受 `accepted:true` 和匹配的 base revision。
- `render_preview`：Rust 渲染，通过 MCP ImageContent 返回真实图像。
- `export_image`：创建异步导出任务。
- `undo_operations`：显式确认后回退上一图，生成新的 revision。
- `detect_landmarks` / `detect_regions`：返回 `browser_provider_required`，实际检测由浏览器 MediaPipe 提供。

所有工具沿用 `ToolResult`，不泄露服务器路径。此网关不运行语言模型；Python DeepAgent 服务选择工具、构造参数并提出提案。

## 完成标准

- [x] Gin + Viper + 官方 MCP SDK 可运行。
- [x] Go 导出使用 Rust 同一核心，删除重复的 Go 像素算法。
- [x] 原图不可变，sourceAssetId 与 session asset 校验。
- [x] 超时、并发限制、异步任务状态和取消。
- [x] 提案、明确接受/拒绝、revision 校验、graph 保存和撤销。
- [x] 真实 Streamable HTTP MCP 工具可由官方 SDK 调用。
- [x] 未实现的检测不返回伪造关键点或蒙版。
- [x] 无效参数返回 422，错误不泄露文件路径。
- [x] 上传删除竞态与 Race Detector 验证通过。

## 测试用例与证据

2026-10-12，在 Rust release Worker 构建后运行：

```text
GIN_MODE=release go test ./...       通过
GIN_MODE=release go test -race ./... 通过
go vet ./...                        通过
```

自动化覆盖：上传、identity、曝光改变像素、裁剪/胶片、旋转/翻转、关闭效果、源 asset 不匹配、Rust 非法参数拒绝、Worker 超时、排队取消、并发上限和临时目录清理。

同格式 no-op 回归覆盖 disabled 坏键与越界值、identity 坏键返回 422，enabled identity multiply 确实改变像素，合法 disabled 仍返回源原字节；缺少 Worker 时只有空图可直通，任何含节点图均返回 503。普通 identity 验收以解码后的像素一致为准，不要求 Rust 与原编码器输出相同压缩字节。

新增真实人像集成用例：同一张 820×1024 JPEG 分别执行全部 22 类操作（几何、网格/液化、脸型/五官/皮肤/身形/衣物、clone/heal/inpaint、画笔、基础颜色/曲线/HSL/胶片/LUT），逐项核对像素或几何实际改变，最后核对源 JPEG 原字节不变。保护蒙版用例逐像素核对左半图片不变且右半实际改变，覆盖 color、liquify、clone、heal、inpaint 和 paint。

提案测试覆盖 HTTP 拒绝状态保存、不修改图与 revision、已拒绝不可应用、已应用不可拒绝、未确认不可应用、过期 revision 拒绝、原生非法参数在提案存储前拒绝，以及“Rust 校验进行期间用户拒绝”不会提交的确定性竞态。

JPEG 方向测试覆盖 Orientation 1..8、越界方向和损坏 EXIF；真实 `portrait-exif6.jpg` 样本原始压缩尺寸为 820×1024，验证上传、AI inspect 与 native PNG 导出均为 1024×820，读取方向不修改源 JPEG 字节。共享样本位于 `services/api/internal/api/testdata/portrait-exif6.jpg`。

官方 MCP SDK HTTP 集成测试覆盖初始化、列出 8 工具、检查图片、生成提案、拒绝未确认操作、确认应用、旧版本冲突、撤销、真实预览 ImageContent 以及浏览器检测能力错误。

独立 API 端口 `18082` 的 curl 验证完成：health、PNG 上传、Rust 曝光输出（85 bytes PNG）、提案生成、明确接受后 revision=2，以及 MCP initialize 返回 `img-editor 0.2.0`。

## 当前进度与限制

本功能 Demo 已可运行。任务与 proposal 只保存在内存，暂不做账号鉴权、持久化队列和云端存储。进度反映执行阶段；尚未接入 Rust 算法内部百分比、实时 GPU 任务调度或 stdio 独立进程。Web 的本地编辑与模型推理使用各自功能文档对账。

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-12 | 将 Go 像素处理替换为 Rust CLI；接入官方 MCP SDK、统一人工/AI事务、任务进度与取消；新增独立提案拒绝与竞态保护，提案保存前原生参数校验；以真实人像覆盖全部 22 类操作和保护蒙版。 |
