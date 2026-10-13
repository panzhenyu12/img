# FUNC-014 背景多余人物与杂物去除

状态：已验证（一期本地 Demo）

关联需求：`REPAIR-004` 背景路人去除、`REPAIR-005` 杂物清理、`REPAIR-006` 主体保护。

## 用户行为

1. 在“修复与仿制”工具中选择矩形、套索或多边形；建议对路人和复杂轮廓使用多边形。
2. 在画布上圈出需要移除的路人、物体或背景瑕疵。
3. 点击“去除多余人物 / 背景杂物”，创建独立的 `repair.inpaint` 操作层。
4. 对主体人物、脸部和衣物点击“保护主体（冻结画笔）”，在画布上涂抹蓝色冻结区域，然后再次预览或导出。
5. 需要指定背景纹理时，仍可使用 Alt 取样和“从取样点修补选区”进行 `repair.clone`。

去除操作是可撤销、可关闭和可删除的编辑层；原图和手工修图链路不依赖 AI。

## Operation 协议

```json
{
  "type": "repair.inpaint",
  "params": {
    "region": { "x": 0.28, "y": 0.20, "width": 0.24, "height": 0.52 },
    "mask": {
      "shape": "polygon",
      "points": [
        { "x": 0.29, "y": 0.21 },
        { "x": 0.50, "y": 0.22 },
        { "x": 0.49, "y": 0.71 },
        { "x": 0.28, "y": 0.70 }
      ],
      "feather": 10,
      "invert": false
    },
    "protectMask": {
      "shape": "brush",
      "strokes": [
        { "x": 0.68, "y": 0.55, "radius": 0.035, "strength": 100 }
      ],
      "feather": 0,
      "invert": false
    },
    "radius": 0.035,
    "iterations": 16
  }
}
```

`mask` 是待移除区域，支持 `rect`、`ellipse`、`polygon` 和画笔编辑；`protectMask` 是主体冻结区域，支持相同蒙版协议。坐标与笔刷半径使用归一化值，`radius` 以短边归一化。Rust 校验限制 `iterations` 为 1–64，笔画与点数受通用上限约束。

## 算法实现

Rust `inpaint.rs` 使用确定性的exemplar纹理补全：

1. confidence×isophote选择填补frontier；
2. masked SSD在原始未选区、未冻结、非透明的完整clean patch中匹配；
3. coarse search与refinement复制匹配块；
4. 小图/窄背景先缩匹配窗，再缩复制patch，仍无donor则报错；
5. iterations是搜索质量预算，不能用少迭代提前终止填洞；冻结区与选区外逐字节不变。

7项自动单测覆盖棋盘/横纵条纹原frequency和phase、保护/透明取样、小图自适应与无donor错误。真实街景release实测1.5–1.9秒，debug/release输出PNG一致；选中人物已移除，放大仍有块状接缝。见 [补全审计](../qa/algorithm-audit-inpaint.md)。

这是一期 Demo 级传统算法，适合背景平面、小型路人和杂物。它不会凭空生成被遮挡的建筑、人物或复杂纹理；大面积遮挡、重复纹理、头发和人与主体严重相交的场景应使用更细的保护/仿制选区，后续再接可选 AI 生成式 inpainting。AI 只能追加或预览操作，不会阻塞手工流程。

## 对账位置

- Web 入口与选区交互：`apps/web/src/components/ToolPanel.tsx`、`apps/web/src/App.tsx`
- Canvas polygon/保护画笔：`apps/web/src/components/CanvasStage.tsx`
- Rust 填补：`core/image-core/src/geometry.rs`
- Operation 路由与校验：`core/image-core/src/engine.rs`、`core/image-core/src/validation.rs`

## 验收标准

- 多边形选区写入 `repair.inpaint.params.mask.shape = polygon`，至少包含 3 个点。
- 选区内背景或杂物像素发生变化，选区外像素保持不变。
- 涂抹冻结后写入 `protectMask.shape = brush`；冻结区域 RGB 与 alpha 逐像素保持源图。
- 保护区域不作为填补 donor，避免主体颜色污染背景。
- 工程保存、撤销、关闭图层和导出均可正常工作。

## 测试用例

- Rust：`polygon_inpaint_removes_selected_object_without_bleeding_protected_subject`，验证 polygon 目标、冻结区域逐像素不变、目标被填补以及 alpha 保持。
- Rust：`selected_inpaint_changes_marked_area_and_preserves_protection`，验证矩形兼容协议。
- Playwright：`多边形去除背景杂物并用冻结笔刷保护主体`，验证真实画布交互、操作参数和预览变化。

## 当前进度

- [x] Web 手工矩形/套索/多边形选区
- [x] 一键创建背景去除操作层
- [x] 主体冻结保护画笔
- [x] Polygon mask 与 protectMask Rust 执行
- [x] WASM/native 共用与自动化验收
- [ ] 语义主体自动分割
- [ ] 生成式复杂背景重建

## 变更记录

- 2026-10-13：新增一期背景人物/杂物去除入口、保护主体流程和 polygon/protectMask 回归测试。
