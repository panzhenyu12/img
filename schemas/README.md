# 编辑协议与 AI Schema

Web 手工编辑、AI 提案、Go API 和 Rust 渲染共享 Operation / EditGraph。Schema 描述编辑对象与业务消息；运行时具体参数由 Rust 校验，MCP 传输由官方 Go SDK 实现。

| 文件 | 作用 |
|---|---|
| `operation.schema.json` | 非破坏编辑操作，含稳定 type、params、controlPoints 与来源信息。 |
| `edit-graph.schema.json` | sourceAssetId 之上的有序 operations。 |
| `mask.schema.json` | inline 选区、效果与保护蒙版，支持矩形、椭圆、多边形、画笔和线性/径向范围。 |
| `edit-proposal.schema.json` | 待审阅的新增操作；baseRevision 是字符串。 |
| `ai-tool.schema.json` | 业务 ToolCall / ToolResult 封装，通过 oneOf 区分调用和结果。 |
| `operation-catalog.json` | 供 AI 查询的参数目录，含结构、单位、范围、默认值与胶片预设；由脚本生成。 |

编辑图示例：

```json
{
  "version": "1.0",
  "sourceAssetId": "uploaded-asset-id",
  "operations": [
    {
      "id": "face-001",
      "type": "face.adjust",
      "version": "1.0",
      "enabled": true,
      "origin": "manual",
      "params": {
        "region": {"x":0.2,"y":0.15,"width":0.6,"height":0.6},
        "adjustments": {"faceWidth":12}
      },
      "controlPoints": []
    }
  ]
}
```

坐标与数值：region、mask、controlPoints 和 strokes 使用 0..1；crop/resize 使用像素，rotate 使用角度，特殊参数按目录单位执行。JPEG width/height 使用 EXIF 校正后的展示方向。蒙版放在 params.mask / params.protectMask；maskId 与 targetId 目前仅为保留元数据，未接资源注册或自动查询。

`body.adjust` 还可携带 `bodyPart`（`full-body`、`torso`、`waist`、`hips`、左右上臂/前臂/大腿/小腿），与浏览器 Pose 生成的 polygon `mask` 和 `landmarks` 一起回放独立部位调整。

operation.schema.json 的 params 接受各算子的扩展字段，不能靠该文件单独完成所有算子范围校验。生成目录也不是完整 JSON Schema 验证器；未知键、类型和范围由 Rust 严格校验，包括提案中的 disabled 操作。

## MCP 的当前契约

Go /mcp 已实现 Streamable HTTP，注册 8 个工具：inspect_asset、propose_operations、apply_operations、render_preview、export_image、undo_operations 六个可用；detect_landmarks 与 detect_regions 不可用，返回 browser_provider_required。当前没有 stdio 入口。

标准 MCP tools/call 的参数为 `{name,arguments}`，嵌入 JSON-RPC params；每个工具的 arguments Schema 由官方 SDK 从 Go toolArguments 生成。ai-tool.schema.json 定义业务封装，不替代 MCP JSON-RPC、SDK tool Schema 或 Rust 参数校验。预览的真实图像在 MCP ImageContent 中，业务 ToolResult 保存结构化元数据。

工具和编辑请求中的 revision / baseRevision 始终使用 string，例如 `"1"`。当前生命周期为 proposed→applied / rejected / expired：提案创建先经 Rust 校验；预览不提交；apply 必须 accepted:true 且版本匹配；reject 使用独立 HTTP `/api/v1/sessions/{id}/proposals/{proposalID}/reject`，不修改图或版本；undo 恢复上一图并产生新版本。HTTP undo 需 baseRevision，MCP undo 还需 accepted:true。

Schema 保留 accepted 状态和 approvalMode 的 auto/confirm/disabled 值。当前 API 固定创建 confirm 提案，未实现 auto 策略或中间 accepted 状态。apply accepted:false 是错误，不是 reject；不能用 Schema 中的预留枚举推断运行时已经实现自动应用。

可选 Python DeepAgent 仅开放 inspect_asset、render_preview、propose_operations；没有 OPENAI_API_KEY 时推荐返回 503。模型不直接覆盖原图，应用与拒绝由用户界面完成。详细流程见 [MCP/AI 集成文档](../doc/mcp-ai-integration.md)。

## 新增操作时的同步要求

1. 更新 Operation Schema type 枚举、Go 白名单和共享类型。
2. 在 Rust 实现参数校验与 WASM/native 图像处理。
3. 更新 Web 参数 registry、面板和参数单位。
4. 运行 `node scripts/generate_catalog.mjs`，生成 schemas/operation-catalog.json、Go 的同名内嵌目录及 doc/func/FUNC-010 参数对账表；不要手工修改生成文件。
5. 在 doc/func/FUNC-xxx-*.md 记录手工/AI 共享参数、可调用示例、验收证据和限制。

版本、来源和可读说明写入 Operation；Go 类型与边界校验位于 services/api/internal/ai/protocol.go。已注册参数不等于效果品质或真实 AI 模型已经验收，完成状态以功能文档和实际测试为准。
