# scene-only 耳饰候选独立复核（v5）

2026-10-15。v5 是冻结 v2 source layers、alpha 与 inverse geometry 的虚拟 scene backing；它没有通过原尺寸耳饰/边缘验收，也不应选择 +100 或 −100 中“看起来更好”的方向上线。

复核脚本：[audit_neural_v5.py](audit_neural_v5.py)；数值报告：[report-v5.json](report-v5.json)。覆盖 128/256 两个 LaMa tile 尺寸和 user 原照 `+100/-100` 两个方向，共 4 行。128 与 256 输出的几何、写域和像素统计相同，仅神经 backing 的背景 tile 误差略有不同。

## 硬约束结果

- 输出 alpha、冻结 inverse map 和写域外像素全部通过；独立用冻结 source alpha/foreground、保存 background 和 inverse map 重建，4 行 max byte error=0。
- 透明金属核心 2,332 像素逐字节保持源图；独立源高光探针（accessories≥.05、min RGB≥160、通道差≤30）共 1,027 像素，4 行均逐字节保持。
- 头发/耳朵 semantic ears 区域没有新增变化；unknown 新覆盖为空。
- 实际几何没有被 scene backing 偷换：+100 inverse SHA `9423e…`，−100 `4770a…`；+100 移动 76,380 像素、最大 12.54 px、移动平均 2.794 px；−100 移动 80,034 像素、最大 11.16 px、移动平均 2.289 px。它保持了完整 V36 几何位移，不能用“减小形变”解释视觉差异。

## 未通过项

raw `accessories≥.05` 语义区域与写入域重叠。+100 两行各有 435 个、−100 两行各有 1,112 个 changed pixels 落在该区域；因此不能称“所有饰品语义像素都保护”。Opaque 核心和高光虽然没动，但软边、孔隙、附着处和模型语义边缘仍被合成改变。该数值不是说 435/1112 个都是真金属，而是说明当前 v5 没有建立足够细的饰品保护合同。

原尺寸组件面板仍能看见：

- +100 收脸：耳饰旁新肤色/暗边与旧轮廓相邻，未形成无断裂连续边。
- −100 扩脸：黑楔较前一轮减轻，但耳钉附着和链条外缘仍有台阶，脸颊外轮廓不自然。

`neural-backing-v5-virtual-scene` 的输出背景由“实际 scene + 最近 scene prior + LaMa”共同产生，仍是隐藏背景假设，不是被遮住内容的观测真值。128/256 的视觉差异不能自动解释为质量提升。

## 结论

v5 证明了 scene-only 版本可以严格保持 opaque metal/highlight、alpha、geometry 和 output write domain 的若干硬约束，但软 accessory overlap 和原尺寸边缘质量仍失败。保持 v3/v4/v5 全部证据，不选择最好看的一个作为默认；继续修复源 matte 连续性、source-clear 与 destination-write 一致性，以及 soft accessory/ear attachment 的保护边界。
