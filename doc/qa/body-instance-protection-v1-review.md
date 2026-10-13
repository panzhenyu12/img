# Body Pose × FASHN instance protection V1

日期：2026-10-13。该候选把 MediaPipe Pose 的实例分割 alpha 作为 FASHN 18 类解析的前景上限，即使只有一个人也启用；它不会替换 FASHN 的衣物类别或骨轴宽度。

## 真实样片证据

`fashion-standing` 的右上臂外侧 `(x=79,y=340)` 是 FASHN `top=255` 的椅背亮线。旧单人路径没有传 Pose instance，partMask 把该像素当袖子；收缩后使用深色椅背供体 `[22,47,49,255]`。启用 instance 后同一行 partMask 从 `[77,167]` 收到 `[104,167]`，该点 alpha 从 `255` 变为 `0`，输出恢复原图亮线 `[105,134,142,255]`，袖内白衬衫仍在目标层。

使用当前生产 renderer、同一调整图和 FASHN 缓存回放五张照片共 15 项：保护区、partMask 外、subjectMask 外和 support 外变化均为 0；输出均有非零编辑。V4 原 renderer 的其余输出作为对照，未发现 Pose instance 导致的可见袖、裙腰或电脑边界丢失。该合同与目视结果只证明样本上的分区改善，不把 15 项自动升格为 8 分。

## 自动验证

- `apps/web/tests/body-parts.spec.ts`：3/3 passed，包含 raster instance 的 outside=0 / inside=255 断言。
- `cd apps/web && npm run build`：passed。
- 生产回放：`BODY_RENDER_BINARY=core/image-core/target/release/image-render BODY_REVIEW_VERSION=v11 BODY_PREVIOUS_VERSION=v4 node scripts/replay_body_parts.mjs`；回放脚本与产品共用 Pose instance 路径。

## 限制

Pose lite 的实例 alpha 仍是低分辨率软边；遮挡严重、多人交叠或手臂与身体粘连时，partMask 仍需用户保护/校正。该修正只收紧 FASHN 解析的前景归属，不解决宽袖内部衣料分层和窄揭示带背景纹理重建。
