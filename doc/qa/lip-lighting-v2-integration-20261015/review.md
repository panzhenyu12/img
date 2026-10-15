# 唇部明亮 V2：本地共享核心接入

日期：2026-10-15。`feature.adjust.adjustments.lipBrightness` 候选 C 已接入现行 Rust native/WASM，本地 Web 与 Go API 已使用新产物。脸型仍为 v19 + physical-fringe-v2，整体脸型、ZIM 整人覆盖及有界 F/B 原型没有接入。

## 改动与实际效果

唇部亮度从加性 sRGB 提亮改为线性 RGB 共同比例调节，正向带色域肩部，负向保留已有中性高光保护。黑色唇纹不再漂灰，饱和红唇能够压暗。分割及唇/内嘴蒙版没有修改。

五张独立照片、55 组冻结对照和等亮度消融见[图像评审](../lip-lighting-v2-20261015/review.md)；[独立评审](../lip-lighting-v2-independent-review-20261015.md)支持只接入该项修复。实际生产 native 与这些照片使用的 candidate C **SHA-256 完全相同**，无需以另一轮重复渲染冒充新证据。

## 回归与部署核验

| 检查 | 结果 |
| --- | --- |
| Rust 完整回归 | 库测试 209 通过、3 忽略；公开操作合同 10 通过；新唇亮度合同 5 通过；0 失败 |
| native / WASM release | `make native wasm` 通过 |
| 真实浏览器 / Go API | 1 项回归内含 0、±30、±100 五档；整张 RGBA 逐字节一致 |
| 原缺陷红绿 | 旧产物在 +30 改灰 649 个黑色像素，回归失败；新产物五档均保留黑色纹理，通过 |
| 图像合同 | 零值恒等、非零确实改变、alpha / 选区外 / 牙齿 / 手工保护保持 |
| Web 生产构建 | TypeScript 与 Vite 构建通过 |
| 本地服务 | Web 5173 与 API 8080 均响应成功；HTTP 实际返回的 WASM、公用目录与 dist 副本 SHA 相同 |

合成浏览器样片包含透明区域，但 Canvas 导入会规范化隐藏 RGB；**隐藏 RGB 的保持由 Rust 公开合同及 CLI 探针验证**，不把浏览器通路误称为完整隐藏 RGB 验收。此次跨端回归调用实际浏览器引擎和 API，没有重复人工拖动页面滑杆，也没有重跑无关 E2E。

| 产物 | SHA-256 |
| --- | --- |
| native / 已审候选 C | `adc18a9e67414e1aac6e554afe9bbb14813229ed9a37f5bd41670b2914cd364a` |
| WASM | `ea036ebf3aa396504e9eb518d9a22b5894432c65d4835999a79eb79221299836` |
| color.rs | `7d7bf5594cf3fb56f0e9b9b406bf9b228562e2db5132192592f5147105dfc597` |

完整哈希、执行命令、服务响应与测试统计在 [validation.json](validation.json)，红绿报告和构建日志保存在本目录。API 每次渲染启动 native，Vite 直接提供新 WASM，因此更新构建产物即可生效；已有网页需刷新以重新加载 Worker/WASM。

## 保留失败

强负值下，浓妆和浅色唇仍可能显露唇缘/牙齿保护边界。高饱和彩色通道的正向提亮余量有限。本次不修改这些蒙版，也不把单项颜色改善记为妆容整体 8 分。瘦脸的耳缘、饰品形变和多人实例归属仍未通过，证据见[当前推进记录](../algorithm-progress-20261015.md)。
