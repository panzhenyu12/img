# 功能实现对账文档

本目录记录每个可交付功能从需求到代码、测试和验收的对应关系，便于产品、设计、前端、后端和算法逐项对账。

## 文档命名

```text
FUNC-编号-功能短名.md
```

编号一旦分配不复用。功能拆分或合并时，在原文档的“变更记录”中说明，并建立关联文档。

## 统一模板

每个功能文档必须包含：

- 功能编号与名称
- 状态（规划中 / 开发中 / 已实现 / 已验证 / 暂停）
- 关联需求 ID（来自 `doc/feature-list.md`）
- 用户行为
- 前端实现
- 后端实现
- Operation 类型
- 参数结构
- 完成标准
- 测试用例
- 当前进度
- 变更记录

## 对账规则

1. 先登记需求 ID，再实现代码。
2. 前后端协议变更必须同时更新 schema 和功能文档。
3. “已实现”表示代码可运行；“已验证”还必须有自动化或手工验收证据。
4. AI 功能只能追加 Operation、蒙版或图层，不能覆盖原图，也不能阻塞手工链路。
5. 每个 Sprint 结束时更新对应文档的当前进度和变更记录。

## 当前功能索引

- [FUNC-000 工程状态](FUNC-000-project-status.md)
- [FUNC-001 编辑器基础链路](FUNC-001-editor-foundation.md)
- [FUNC-002 图片导入与临时渲染](FUNC-002-image-upload.md)
- [FUNC-003 Rust 图像核心](FUNC-003-rust-image-core.md)
- [FUNC-004 AI / MCP 工具调用桥](FUNC-004-mcp-ai-tool-bridge.md)
- [FUNC-005 基础调色与胶片效果核心](FUNC-005-color-and-film-core.md)
- [FUNC-006 基础图形变换](FUNC-006-geometric-transforms.md)
- [FUNC-007 共享 Rust WASM/native 渲染引擎](FUNC-007-shared-rust-render-engine.md)
- [FUNC-008 Web 手工编辑工具](FUNC-008-web-editor-tools.md)
- [FUNC-009 Go 渲染、任务与 MCP](FUNC-009-go-render-mcp.md)
- [FUNC-010 116项参数逐项对账](FUNC-010-parameter-ledger.md)
- [FUNC-011 人脸轮廓变形质量与验收](FUNC-011-face-contour-quality.md)
- [FUNC-012 身体轮廓分层重塑（真实图仍未达8分）](FUNC-012-body-semantic-masks.md)
- [FUNC-013 衣物去褶皱与纹理保护](FUNC-013-clothing-wrinkle-retouch.md)
- [FUNC-014 背景多余人物与杂物去除](FUNC-014-background-person-removal.md)
- [FUNC-015 皮肤、肤色与妆容（48项肤质/肤色检查，强磨皮自然度失败）](FUNC-015-skin-and-makeup-quality.md)
- [FUNC-016 客户端与本地服务端 LaMa 填补](FUNC-016-local-model-inpaint.md)
- [FUNC-017 蒙版效率与身体滑杆缓存](FUNC-017-render-efficiency-and-cache.md)
- [FUNC-018 手动瑕疵修复与光照匹配（小范围约7.8）](FUNC-018-manual-heal-quality.md)
- [人像与衣物修图算法调研](../research/portrait-retouching-algorithms.md)
- [一期验收证据](../qa/phase1-validation.md)
- [当前工程验证与已知失败](../qa/current-validation.md)
- [皮肤/肤色实拍复核与妆容证据边界](../qa/skin-guided-review.md)
- [AI / MCP 集成方案](../mcp-ai-integration.md)

一期实际交付以 [开发计划](../phase1-development-plan.md)、[FUNC-000](FUNC-000-project-status.md) 及当前验证记录为准。完整功能清单记录长期需求。当前已接入 MediaPipe/Pose、BiSeNet 皮肤/头发等像素类别解析、独立前景轮廓路径及本地 LaMa；这些实现与工程通路通过不代表逐功能8分完成。

皮肤48项仅验证肤质/肤色支持域与alpha，并有强磨皮蜡感、浅肤色层次与色偏失败；缓存缺新 accessories/ears 蒙版，唇/眉/牙齿妆容的独立证据仍不完整。身体宽衣/遮挡部位、轮廓软边、窄带纹理与复杂补全质量仍未全部达标；完整身体部位/服装解析、眼袋/泪沟结构、GPU及原生客户端仍属于后续目标。
