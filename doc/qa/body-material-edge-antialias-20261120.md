# 身形材料边界抗锯齿回归（2026-11-20）

## 范围

输入为 `test/results/body-fashion-source-20261119/fashion-standing/source.png` 的 native `waist=-30` 回放。该回放沿用已冻结的检测结果，未重新推理模型；输出仅写入 `/tmp/fashion-soft-oldmask-30.png`。

## 结果

- 新增 2 个 Rust 单测均通过：材料边界产生受限的 fractional coverage，且不会超过 subject ownership。
- 重新编译 `image-render` 成功。
- 原图尺寸与 alpha 保持；保护层合同未改动。
- 原尺寸局部审图显示白衬衣/深色外套交界的硬阶梯减弱，但旧 part mask 的实例归属仍会留下少量边界残差，因此本项不能单独宣称身形整体可用。

## 后续

需要把前端 `deriveBodyLayers` 产生的新 torso ownership 写回 fresh graph，再复测同一 `waist=±30`，确认解析器归属修复和 Rust coverage 过渡叠加后不再出现可见断角。
