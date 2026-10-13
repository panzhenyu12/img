# AI / MCP 图像编辑集成

版本：v0.2，2026-10-12。本文记录当前 Demo 的真实接口和能力；完整接口见 [Go API 文档](../services/api/README.md)，实现与验证见 [FUNC-009](func/FUNC-009-go-render-mcp.md)。

人工和 AI 共用 `Operation` / `EditGraph`，都由 Rust 图像核心执行。AI 理解意图、选择已有能力并提出参数，浏览器负责预览和确认。AI 服务、模型或网络不可用时，本地手工编辑、Rust/WASM 和浏览器导出继续可用。

## 当前运行结构

```text
Web 手工面板 / AI 提案预览
  → Operation / EditGraph
  → Web Worker → Rust/WASM → Canvas 2D

Web 自然语言请求
  → 可选 Python DeepAgent 服务 POST /recommend
  → MCP Client → Go /mcp（Streamable HTTP）
      → inspect_asset：当前图和参数目录
      → render_preview：真实图像
      → propose_operations：经过 Rust 校验的提案

Web 接受 / 拒绝提案
  → Go HTTP 编辑事务 → 新 revision / 提案状态

Go 预览 / 导出
  → Rust native image-render Worker → PNG / JPEG
```

WASM 是图像算法运行时，MCP 是宿主提供的工具协议。浏览器 WASM 通过 JS 执行编辑；当前 Go MCP 通过 Rust native 子进程执行预览与导出。浏览器不监听 MCP 端口，Go 也未嵌入 WASM runtime。

Go 已接入官方 `github.com/modelcontextprotocol/go-sdk v1.8.0`，服务名为 `img-editor`，版本 `0.2.0`。MCP 地址为 `http://127.0.0.1:8080/mcp`，实现 Streamable HTTP 的 initialize、tools/list 和 tools/call。当前没有 stdio MCP 启动入口、独立 stdio 进程或浏览器远程执行桥。

## 已注册的 8 个 MCP 工具

6 个工具可执行；2 个检测工具保留在目录中并声明 unavailable。

| 工具 | 可用 | 必要业务参数（均需 sessionId） | 当前行为 |
|---|---|---|---|
| `inspect_asset` | 是 | 无额外参数 | 返回 asset、字符串 revision、editGraph、Operation 参数目录、工具目录和坐标说明；不推理图片内容或人物数量。 |
| `propose_operations` | 是 | baseRevision、operations | Rust 验证与当前图合并后的操作，再保存 proposed 提案；返回 pending_approval 和 proposal。工具本身不运行语言模型。 |
| `apply_operations` | 是 | proposalId、baseRevision、accepted:true | 验证完整图、核对版本与提案状态后追加操作，保存上一图并产生新 revision。 |
| `render_preview` | 是 | 可选 editGraph、format、quality | 渲染指定图或当前图；返回尺寸、格式等元数据，以及真正的 MCP ImageContent。 |
| `export_image` | 是 | 可选 editGraph、format、quality | 创建异步 Rust 导出任务，返回 job 和 statusUrl；不修改编辑图。 |
| `undo_operations` | 是 | baseRevision、accepted:true | 恢复上一份编辑图，并产生新 revision；版本号不会回退。 |
| `detect_landmarks` | 否 | 无额外参数 | 返回 browser_provider_required，不返回伪造关键点。 |
| `detect_regions` | 否 | 无额外参数 | 返回 browser_provider_required，不返回伪造人物、皮肤或衣物蒙版。 |

`tools/list` 的工具输入 Schema 由官方 SDK 根据 Go `toolArguments` 生成。上述业务必填字段还由执行器核对；Operation 参数目录通过 `inspect_asset.result.operations` 或 `GET /api/v1/operations` 提供。参数类型、未知键和取值范围最终由 Rust 校验，提案创建也校验 disabled 操作。

浏览器已有本地 MediaPipe Face/Pose 检测，可帮助用户定位区域；它没有接入这两个服务端 MCP 检测工具。当前未实现服务端检测推理或人物/皮肤/衣物分割。

## MCP 消息与普通 HTTP 的区别

MCP 客户端发送标准 JSON-RPC tools/call，例如：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "inspect_asset",
    "arguments": {"sessionId": "uploaded-session-id"}
  }
}
```

业务结果使用 ToolResult，放在 MCP structuredContent 和 TextContent 中；工具失败设置 `isError:true`，并返回 `status:"error"` 与 error.code/message。render_preview 还返回单独的 ImageContent，PNG/JPEG 字节由 MCP 传输编码。inspect_asset 只有元数据和编辑上下文；需要看图时应调用 render_preview，并使用支持图像输入的模型与适配器。

[ai-tool.schema.json](../schemas/ai-tool.schema.json) 是业务调用/结果封装，不是整个 MCP JSON-RPC 协议，也不是每个工具的完整输入 Schema。普通 `POST /api/v1/tools/{name}/call` 接收 arguments 对象并返回业务 ToolResult；其 render_preview 响应只有元数据，图像应通过 MCP ImageContent 或 HTTP `/sessions/{id}/preview` 获取。

## 编辑版本与提案状态

revision 和 baseRevision 始终为字符串，例如 `"1"`，不可转换为 number 后发送。它们表示当前 session 的编辑版本，不是文件路径或图片 hash。

当前真实状态流为：

```text
proposed → applied   用户接受、版本匹配且 Rust 校验成功
proposed → rejected  用户调用独立 reject 接口
proposed → expired   尝试应用时发现 baseRevision 已过期
```

Schema 中保留了 accepted 状态以及 approvalMode 的 auto/disabled 枚举，当前 API 创建提案固定为 `approvalMode:"confirm"`，未实现自动应用策略，也不会保存中间 accepted 状态。`accepted:true` 是应用或 MCP 撤销请求中的确认字段。

创建提案和预览不修改编辑图、不增加 revision。接受会追加 Operation 并增加 revision；拒绝只修改提案状态，图和 revision 均不变。apply 省略 accepted 或传 false 会返回错误，不代表拒绝。已拒绝、已应用和已过期的提案不能应用；已应用提案不能被 reject。

应用过程中如果用户拒绝，Rust 返回后会再次核对提案状态并阻止提交；如果编辑版本改变，也会阻止提交。原图不覆盖，保存人工编辑图、应用和撤销都操作同一份 EditGraph。

## Web 与 Go 的协作流程

1. 浏览器导入后在本地编辑；需要 AI 时创建并上传到 Go 临时 session。
2. 推荐前读取当前 session，将本地 EditGraph 通过 PUT graph 与 baseRevision 保存到 Go。
3. DeepAgent 读取上下文、按需查看真实预览，并调用 propose_operations。
4. Web 读取提案列表，把提案 Operation 临时叠加到本地图进行 Rust/WASM 预览。临时预览不写历史、工程或导出。
5. 用户接受时核对本地编辑与服务端基础图，并调用 apply；拒绝时使用独立 reject。
6. 接受后把返回的完整 editGraph 写入本地历史；浏览器可以撤销，也可经后端 undo 恢复上一图。

推荐或确认期间发生手工编辑时，Web 检查图的指纹并提示重新生成，避免旧提案覆盖当前手工结果。后端进一步用 revision 处理并发冲突。

以下路径统一加 `/api/v1`；表中使用相对于该前缀的路径。

| 方法 | 路径 | 请求 | 结果 |
|---|---|---|---|
| GET | `/sessions/{id}` | — | `{id,image,revision,editGraph}` |
| PUT | `/sessions/{id}/graph` | `{baseRevision,editGraph}` | `{revision,editGraph}` |
| POST | `/sessions/{id}/operations` | `{baseRevision,operations}` | 追加人工操作，返回 `{revision,editGraph}` |
| POST | `/sessions/{id}/proposals` | `{baseRevision,operations,summary}` | 201 EditProposal |
| GET | `/sessions/{id}/proposals` | — | `{proposals:[...]}` |
| POST | `/sessions/{id}/proposals/{proposalID}/apply` | `{baseRevision,accepted:true}` | `{revision,editGraph}` |
| POST | `/sessions/{id}/proposals/{proposalID}/reject` | `{baseRevision}` | `{proposal,revision,editGraph}` |
| POST | `/sessions/{id}/undo` | `{baseRevision}` | `{revision,editGraph}` |
| POST | `/sessions/{id}/preview` | `{editGraph,format,quality}` | PNG/JPEG 字节，不提交图 |
| POST | `/sessions/{id}/render` | `{editGraph,format,quality}` | PNG/JPEG 字节 |
| POST | `/jobs` | `{sessionId,editGraph,format,quality}` | 202 任务状态 |
| GET | `/jobs/{jobID}` | — | 状态、阶段进度和成功后的 resultUrl |
| POST | `/jobs/{jobID}/cancel` | — | 取消排队或执行中的任务 |

HTTP undo 由用户界面发起，只需 baseRevision；MCP undo 还需 accepted:true。reject 是 HTTP 端点，不是第 9 个 MCP 工具。HTTP 编辑事务失败返回明确错误，通常为 409；渲染非法参数返回 422。详细错误与上传契约见 Go API 文档。

## Operation 参数示例

以下参数可以同时来自手工面板或 AI 提案，具体范围以生成的 [operation-catalog.json](../schemas/operation-catalog.json) 为准：

```json
{
  "id": "face-width-001",
  "type": "face.adjust",
  "version": "1.0",
  "enabled": true,
  "origin": "ai",
  "label": "轻微调整脸部宽度",
  "provenance": {"actorType":"ai","toolName":"propose_operations"},
  "params": {
    "region": {"x":0.2,"y":0.15,"width":0.6,"height":0.6},
    "adjustments": {"faceWidth":12,"jaw":8}
  },
  "controlPoints": []
}
```

脸型、五官、身形和衣物的语义参数放在 params.adjustments；磨皮等操作按目录使用自己的参数。区域、蒙版、控制点和画笔坐标为 0..1；crop/resize 使用像素，rotate 使用角度，其余值按目录单位执行，不把所有数字都当百分比。JPEG 尺寸以 EXIF 方向校正后的展示方向为准。

局部效果使用 params.mask，保护区域使用 params.protectMask，都是 inline mask；透明度和混合模式使用 opacity / blendMode。当前不能只传 maskId 并期待服务端自动查找蒙版；targetId、maskId 是保留元数据，不是已实现的资源注册服务。

propose_operations 只提交新增 operations，不替换整张图。预览提案时先将它们追加到 inspect 返回的 editGraph，然后把完整图交给 render_preview；该工具没有 proposalId 自动组图参数。

## 可选 DeepAgent 服务

[Python 服务](../services/agent/README.md) 通过 `make agent` 启动，默认地址 `http://127.0.0.1:8090`。它使用 DeepAgents、LangChain MCP adapter 和 ChatOpenAI，图像工具仅开放 inspect_asset、render_preview、propose_operations；应用、撤销和导出不交给模型自动执行。

```text
POST /recommend
{"sessionId":"uploaded-session-id","prompt":"轻微调整脸部宽度，使用低饱和胶片色调"}
```

配置项：OPENAI_API_KEY；可选 OPENAI_BASE_URL、IMG_AI_MODEL（默认 gpt-4.1-mini）、IMG_MCP_URL（默认 Go /mcp 地址）。密钥从服务端环境变量读取，不放进 Web 构建、Operation 或工程 JSON。

无 key 时 /recommend 返回 503，提示继续手工编辑；模型任务超过 90 秒返回 504，其他被捕获的推荐失败返回 502。GET /healthz 的 configured 只表示 key 是否存在，不验证模型连通性。推荐接口返回 `{status,message,sessionId}`，Web 再读取 Go 提案列表，不能把模型文字直接当作已应用操作。

已开放真实 MCP 图像预览，但没有配置 key 时，真实模型的图像理解、选参数和推荐质量仍未验收。当前没有生成式去人/去物模型；repair.inpaint 是可手工或通过 Operation 调用的确定性邻域填补，不等同于模型生成式修复。

## 新功能的 AI 接入要求

新增手工能力时同步完成以下工作：

1. 注册稳定的 Operation.type，并更新共享 JSON Schema 与 Go 白名单。
2. 在 Rust 同时支持 WASM 和 native 执行，严格校验参数，拒绝未知键和非法值。
3. Web 面板只构造同一份 Operation，不另建 AI 专用像素实现。
4. 在 Web 参数 registry 与生成脚本中提供范围、默认值、单位和结构；运行生成脚本后同步 schemas 与 Go 内嵌目录。
5. 在 doc/func/ 记录真实效果、自然语言到参数的示例、验收证据和限制。

无需为每个滑杆新增 MCP 工具，模型通过 inspect 获取目录，再提交同样的 Operation。服务端未实现的检测或生成能力必须明确报告 unavailable。

## 当前验证与限制

已有官方 Go MCP SDK HTTP 集成测试覆盖初始化、8 工具列表、inspect、提案、未确认拒绝、确认应用、版本冲突、撤销、可解码的 ImageContent 和检测能力错误。额外 HTTP 测试覆盖独立 reject、拒绝与应用竞态、提案原生参数校验和原图不变，证据见 FUNC-009。

当前 session、proposal、历史和 job 状态均为 Demo 内存数据；未做账号鉴权、持久化、跨客户端同步、自动应用策略、stdio 传输或服务端检测。浏览器本地能力与可选 AI 服务分别验收，不以工具已注册代表模型效果已通过。
