# FUNC-013 衣物去褶皱与纹理保护

状态：衣物细褶纹理弱化与局部调色已实现；大褶皱和版型重建未完成。

关联需求：`CLOTH-001` 衣物局部选区、`CLOTH-002` 褶皱平整、`CLOTH-003` 纹理保留。

衣物操作使用 `clothing.adjust`。用户可以手工矩形、椭圆、套索或多边形标记衣物区域，也可以把人体检测出的区域作为候选选区。没有 AI 或模型时，手工流程仍完整可用。

## 参数与实现

| 参数 | 范围 | 作用 |
|---|---:|---|
| `smooth` | 0–100 | 通过局部平滑弱化细褶纹理；大折痕、衣物形状仍保留 |
| `texture` | 0–100 | 保留织物高频纹理、缝线和印花边缘；数值越高保留越多 |
| `hue` / `saturation` / `brightness` | -100–100 | 仅在选区内调整衣物颜色 |
| `fit` / `edges` / `sleeves` / `skirt` / `collar` / `cuffs` / `sleeveLength` | -100–100 | 未完成专用算法，UI隐藏且非零调用拒绝 |

Rust `color::clothing_retouch` 将输入分为模糊的低频平面和原图高频残差，再按 `smooth`、`texture` 重组。强度从identity连续变化；模糊donor仅来自选区内、未冻结、有效alpha像素，避免区域外颜色污染。最终仍经 `mask::composite` 与局部蒙版合成，衣物选区外的背景像素保持不变。

这是传统确定性细纹弱化算法，不能以梯度下降或像素变化证明大褶皱消失。严重遮挡、破损或需要凭空补纹理的区域应使用后续 AI inpainting；AI 失败不影响手工平滑、修复和仿制图章。

## 对账位置

- Web 区域和参数面板：`apps/web/src/components/ToolPanel.tsx`
- 图层创建与蒙版：`apps/web/src/App.tsx`
- Rust 渲染：`core/image-core/src/color.rs`、`core/image-core/src/geometry.rs`
- 统一操作校验：`core/image-core/src/validation.rs`

## 验收

- `smooth > 0` 会改变衣物选区像素，并保留 alpha。
- `texture=100` 的织物高频差异不低于低纹理设置。
- 蒙版外背景逐像素保持原图。
- Web WASM 与 Go/native 使用同一 Rust 实现。
- `cargo test --manifest-path core/image-core/Cargo.toml`、`npm run build` 通过。

## 当前限制

一期没有衣物语义分割模型，自动人体区域只是候选 polygon；宽松衣服、遮挡和复杂图案需要手工套索修正。频率分离不能生成缺失纹理，也不会把衣服替换为另一种服装。

多图实际效果见 [衣物对比](../qa/clothing-multisample-review.md)：三张照片的局部调色与纹理参数均有变化，选区外逐像素不变；T-shirt、斜衬衫和男衬衫的大折形状仍保留。不能称为几何去褶皱。
