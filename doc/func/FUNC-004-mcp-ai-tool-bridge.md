# FUNC-004 AI / MCP 工具调用桥

状态：MCP/提案链路已验证；真实语言模型推荐待配置后验证

关联需求：CORE-007、CORE-014、LAYER-016（可撤销参数结果）；FACE-027/FILM-022 为可选模型入口；不将 FACE-026 或皮肤分割列为 server 已实现。

## 用户行为与实现

用户可发出修图指令，模型通过 MCP 检查当前工程与能力目录、获取预览图并提交操作提案。Web 临时预览不写入工程；接受后追加普通节点，拒绝不改变原图。生成后出现新手工编辑时，接受按钮必须拒绝旧结果覆盖新编辑。

前端：AIPanel、App 与工程 fingerprint；后端：官方 MCP Go SDK `/mcp`、tools.go、revision/提案状态机；可选 Python DeepAgent：`services/agent/main.py`。详细后端对账见 [FUNC-009](FUNC-009-go-render-mcp.md)。

## 工具与参数

inspect_asset、propose_operations、apply_operations、render_preview、export_image、undo_operations 可用。detect_landmarks/detect_regions 在 server 明确返回 unavailable，Web 有独立 Face/Pose 检测。

提案：`{baseRevision,operations,summary}`；应用：`{baseRevision,accepted:true}`；拒绝独立 `/proposals/{id}/reject`。Operation 包含 origin/provenance；Rust 校验后才保存和应用，提交时再次检查 revision 与提案状态。

## 完成标准与测试

官方 SDK initialize/tools/list/tools/call 真调用通过；MCP预览返回可解码ImageContent；revision冲突、未批准、拒绝/应用竞态、非法参数、undo均有测试。Playwright 覆盖预览、接受、拒绝、撤销和手工修改保护；Python无key时health正常、recommend503，不阻塞编辑。

## 当前限制与变更

只有 Streamable HTTP，尚无 stdio CLI。Agent 已开放只读预览供支持图像的模型使用，但没有密钥时未验证实际模型审美/推荐质量；没有生成式去人、衣物分割。2026-10-12 从 transport-neutral 骨架升级为真实 MCP transport、native工具、Web提案与可选DeepAgent。
