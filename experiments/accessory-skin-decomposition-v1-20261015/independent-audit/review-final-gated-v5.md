# final-gated-v5 耳饰候选独立复核

作者报告明确 `accepted=false`，结论为“仍有 doubled/broken edge”。独立复核保留该拒绝，未选择 +100 或 −100，也没有把 128/256 中更顺眼的一行设为默认。

## 通过的硬约束

- 四行（128/256 × +100/−100）都只在 `final-write` 内改变，unknown/insufficient 写入为 0。
- 输出 alpha、opaque metal core 和 source inverse map 保持。独立冻结位移统计与 v5 相同：+100 移动 76,380 像素、最大 12.54 px、均值 2.794 px；−100 移动 80,034 像素、最大 11.16 px、均值 2.289 px。
- 独立源高光探针（accessories≥.05、min RGB≥160、通道差≤30）1,027 像素，四行均逐字节保留。
- ears semantic 区域和五官语义区域没有新增变化；写域外未变。

## 未通过的证据

旧 `accessories≥.05` 语义区域内仍有 +100 的216、−100 的34 changed pixels。它们属于 strict old-strip gate 允许替换的 source support，并非 opaque metal/highlight；这说明软边、孔隙和耳饰附着附近仍没有独立的物理归属真值。金属核心未变不能推出链条边缘、连接处和旁侧肤色都正确。

原尺寸面板 `final-gated-v5/256/{100,-100}/{left,right}-1x.png` 与 face panels 显示：+100 仍有肤色/暗边双轮廓；−100 只保留 34 个变化，视觉上更接近 v2，但耳饰附着和脸颊边缘的连续性仍不足。两者都是同一冻结几何，不允许通过降低实际运动来换取外观。

结论保持拒绝：该分支可作为严格失败/边缘写域研究证据，不能接入生产或宣称达到8分。需要解决 source support 连续性和软饰品附着，再做独立原尺寸对照。
