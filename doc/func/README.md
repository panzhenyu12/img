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
- [FUNC-011 自研RBF/SVF、可行轮廓和颈连接历史实验](FUNC-011-face-rbf-neck-experiments.md)
- [FUNC-012 身体轮廓分层重塑（真实图仍未达8分）](FUNC-012-body-semantic-masks.md)
- [FUNC-013 衣物去褶皱与纹理保护](FUNC-013-clothing-wrinkle-retouch.md)
- [FUNC-014 背景多余人物与杂物去除](FUNC-014-background-person-removal.md)
- [FUNC-015 皮肤、肤色与妆容（唇亮度V2已接入；四项妆容真实照片回放已记录，整体仍未达8分）](FUNC-015-skin-and-makeup-quality.md)
- [FUNC-016 客户端与本地服务端 LaMa 填补](FUNC-016-local-model-inpaint.md)
- [FUNC-017 蒙版效率与身体滑杆缓存](FUNC-017-render-efficiency-and-cache.md)
- [FUNC-018 手动瑕疵修复与光照匹配（小范围约7.8）](FUNC-018-manual-heal-quality.md)
- [FUNC-020 五官连续形变与眉毛眼睑保护（几何回归通过，自然度/性能未8分）](FUNC-020-feature-svf-quality.md)
- [FUNC-021 分部位轮廓与形状保持（研发中，当前审美效果未通过）](FUNC-021-anatomical-curve-quality.md)
- [FUNC-023 人像边缘颜色分解与遮挡合成（隔离实验，未验收）](FUNC-023-portrait-layer-matting.md)
- [FUNC-024 整体脸型与可编辑预设（隔离原型，未上线）](FUNC-024-whole-face-shape.md)
- [FUNC-025 修图工具→细项→算法→验收对账](FUNC-025-tool-algorithm-ledger.md)
- [FUNC-027 小耳圈细尺度解析与随动保形（隔离算法，未接入）](FUNC-027-small-accessory-material.md)
- [FUNC-028 部位语义目标与连续形变实验（隔离算法，强档仍有失败）](FUNC-028-semantic-deformation-20261022.md)
- [FUNC-030 人像材质归属（分类器隔离，源颈缘修复进入 Rust）](FUNC-030-face-ownership-candidate.md)
- [FUNC-031 五官高度驱动的目标曲线（隔离回归通过，未获实片自然度收益）](FUNC-031-anatomical-segmented-profile.md)
- [FUNC-022 唇周附着与局部应变（隔离组件改善，未上线）](FUNC-022-perioral-attachment-quality.md)
- [人像与衣物修图算法调研](../research/portrait-retouching-algorithms.md)
- [像素蛋糕桌面手册：84组功能与算法映射](../research/pixelcake-manual-functional-audit-20261015.md)
- [一期验收证据](../qa/phase1-validation.md)
- [当前工程验证与已知失败](../qa/current-validation.md)
- [皮肤/肤色实拍复核与妆容证据边界](../qa/skin-guided-review.md)
- [AI / MCP 集成方案](../mcp-ai-integration.md)

一期实际交付以 [开发计划](../phase1-development-plan.md)、[FUNC-000](FUNC-000-project-status.md) 及当前验证记录为准。完整功能清单记录长期需求。当前已接入 MediaPipe/Pose、BiSeNet 皮肤/头发等像素类别解析、独立前景轮廓路径及本地 LaMa；这些实现与工程通路通过不代表逐功能8分完成。

皮肤48项仅验证肤质/肤色支持域与alpha，并有强磨皮蜡感、浅肤色层次与色偏失败；缓存缺新 accessories/ears 蒙版，唇/眉/牙齿妆容的独立证据仍不完整。身体宽衣/遮挡部位、轮廓软边、窄带纹理与复杂补全质量仍未全部达标；完整身体部位/服装解析、眼袋/泪沟结构、GPU及原生客户端仍属于后续目标。

- [FUNC-032 人像速度场过冲与固定材质边界修复（Rust 默认，Web 未重建）](FUNC-032-positive-portrait-field.md)
- [FUNC-033 额头高度与内部发际线联动（Rust 算法，本批单项改善）](FUNC-033-interior-hairline.md)
- [FUNC-035 耳朵、耳饰细尺度解析（解析修正，长耳链漏检仍未解决）](FUNC-035-face-detail-parser-scales.md)
- [FUNC-036 源边缘背景误差与连续底图求解（局部接缝修正，整体未验收）](FUNC-036-source-scene-edge-evidence.md)
- [FUNC-037 稳定源底图与预乘前景配对（修复小位移色差，整体未验收）](FUNC-037-source-layer-pairing.md)
- [FUNC-038 独立衣物/颈部源归属（核心显式证据路径，Web 尚未产生证据）](FUNC-038-source-material-ownership.md)
- [FUNC-039 固定源光学估计域（消除计算留白引起的色差，整体未验收）](FUNC-039-source-optical-domain.md)
- [FUNC-040 完整源颈域与材质边界（隔离算法，修复裁切依赖和软材料串联）](FUNC-040-complete-neck-domain.md)
- [FUNC-041 真实轮廓转角平滑（隔离算法，保留强度，须核实际执行误差）](FUNC-041-face-curve-fairing.md)
- [FUNC-042 耦合形变响应与真实材质边界（隔离算法，大档、小档和下巴组合已定向验证）](FUNC-042-face-response-feedback.md)
- [FUNC-043 背景颜色中心与透明度证据分离（Rust默认源配对路径，旧残线仍未完全解决）](FUNC-043-scene-evidence-roles.md)
- [FUNC-044 额头采样保护与观察头部域（Rust默认，小档端点响应改善，部分强档仍受限）](FUNC-044-hairline-sampling-support.md)
- [FUNC-045 固定边界控制的源身份（native跨零稳定修复，整体瘦脸未验收）](FUNC-045-source-fixed-boundary-controls.md)
- [FUNC-046 源层角色与后发可见性（隔离机制及单张实拍，光学与自然度尚未验收）](FUNC-046-source-layer-role-visibility.md)
- [FUNC-047 源照片光学边缘估计（两项根因已复现，有界颜色候选仍未通过）](FUNC-047-source-optical-edge-estimation.md)
- [FUNC-048 眼睑闭合与眼球材质覆盖（隔离候选，光学边界及开眼未解决）](FUNC-048-eye-aperture-layer-closure.md)
- [FUNC-049 大图人像速度场亚像素精度（Rust 数值修复，瘦脸整体仍未验收）](FUNC-049-face-flow-precision.md)
- [FUNC-050 脸颊与邻发共同运动（隔离算法，外缘保护及曲线可达性回归）](FUNC-050-shared-cheek-material.md)
- [FUNC-051 可动耳廓的源背景权限（禁止供体与隐藏背景重建分离）](FUNC-051-ear-background-permissions.md)
- [FUNC-052 源照片 F/B 有界局部求解（数值修复，亮脊仍未验收）](FUNC-052-bounded-matting-solver.md)
- [FUNC-053 饰品识别与可动皮肤覆盖一致性（隔离源算法，用户图残线改善）](FUNC-053-consistent-portrait-source-evidence.md)
- [FUNC-054 源场景先验与照片边缘残差（原生隔离算法，耳外亮脊改善）](FUNC-054-source-scene-contrast.md)
- [FUNC-055 耳饰随动与共享皮肤场（常用档局部可用证据，强档未通过）](FUNC-055-shared-accessory-motion.md)
- [FUNC-056 手工纠正饰品源类别（深色人像实片验证，自动识别仍失败）](FUNC-056-manual-accessory-source-correction.md)
- [FUNC-057 邻发共同域多图核对（常用档审查，耳饰证据缺口保留）](FUNC-057-neighbor-hair-common-domain-audit.md)
- [FUNC-058 源耳饰约束与耳颊材质（修复支持域跳变，共旋按源域单独验收）](FUNC-058-source-attachment-domain-and-ear-material.md)
- [FUNC-059 脸颊与邻发连续材料域（修复内部误固定，常用档实片候选）](FUNC-059-connected-face-hair-domain.md)
- [FUNC-060 受约束轮廓残差耦合（过冲反例已修，视觉收益仍有限）](FUNC-060-contour-residual-coupling.md)
- [FUNC-061 有界场精确固定自由度消元（同目标、同约束的可选加速）](FUNC-061-exact-fixed-dof-reduction.md)
- [FUNC-062 同版脸宽控制域与源材料证据（三图源审计通过，成片未验收）](FUNC-062-consistent-face-detail-evidence.md)
- [FUNC-063 同源运动界面（修复脸侧固定岛，真实双向与耳廓保形复核）](FUNC-063-source-motion-interface.md)
- [FUNC-064 源材质边界合同（弱耳孤岛及原生采样精度修复）](FUNC-064-source-role-boundary-contract.md)
- [FUNC-065 完整头部光学层（修复多尺度硬背景误判造成的耳外点链）](FUNC-065-head-optical-source-boundary.md)
- [FUNC-067 连续衣料光学层与自动源流程（裸臂单图±30初步可用，袖衣分层与速度待完成）](FUNC-067-body-material-source-layers.md)
- [FUNC-069 自适应真实过渡域候选（隔离回放，dark 转角改善但请求强度/邻点跳变未达晋升条件）](FUNC-069-adaptive-contour-transition-domain.md)
- [FUNC-070 源材质统一运动域候选（隔离回放，默认路径未替换）](FUNC-070-source-material-motion-domain.md)

- [FUNC-074 源虹膜测量约束的眼宽公共调用](FUNC-074-measured-eye-width-public-contract.md)
- [FUNC-075 Rust 内存前后材质分层合成](FUNC-075-body-layered-compositor.md)
- [FUNC-076 源轮廓混色与背景上下文配对修正（隔离实拍候选）](FUNC-076-source-optical-context-repair.md)
- [FUNC-077 显式字节量化源重建（普通Rust入口，瑜伽轻度收腰验证）](FUNC-077-quantized-source-reconstruction.md)
- [FUNC-078 眼部源材料人工修正（独立算法，自动识别失败仍保留）](FUNC-078-eye-source-material-correction.md)
- [FUNC-079 普通 Rust 眼部轻闭合（源准备一次、整数轻档候选，迁移待验收）](FUNC-079-native-eye-closure.md)
