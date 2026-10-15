# 混色旧边缘的背景补全：独立复核

2026-10-15。仅 woman +100，冻结 B（皮肤在头发前）假设、source alpha、前景颜色与几何。局部橙色旧边有改善，但完整画质不通过，B 深度假设也未经自动证明。未上线。

## 当前证据

`mixed-edge/virtual-context/left-1x.png` 比 previous B 减少左脸原轮廓附近的橙色细线；仍有较宽暗带和底端点状细线。`observed-only` 留下较明显青绿色边。独立补充右脸原尺寸面板 `right-comparison-1x.png`，耳根附近仍有硬边/尖突。

脚本 `check_mixed_edge.py` 不调用神经模型，独立使用 SciPy bilinear 从冻结数据重建已变形的预乘前景和 alpha，再读取保存的 background 输出合成。两组最终 RGB 均逐字节复现，max byte error=0；输出 alpha 和 union 域外均不变。因此可以支持“最终合成只改变背景贡献”，不能解释为新的人脸形变算法或源 alpha 改善。

目标区域恰等于原背景 write target（9,482 像素）与 B-write（12,328 像素）的并集，共 15,515 像素。没有扩大到并集外，但并集本身不自动满足所有保护合同。

## 独立发现：原并集包含软耳边

按 ears/accessories≥0.05 独立核对，observed-only 比 previous B 新改了 37 像素，virtual-context 新改了 38 像素，位于右耳侧；头发高置信核心均 0 新变化。源头是原背景 target 仅遵循旧 frozen 门槛，而 B-write 的更严保护没有约束 union 中来自旧 target 的像素。

严格失败脚本与日志保留为 `check_mixed_edge-strict-failure.py` / `strict-failure.log`。量化报告 `report.json` 含绝对坐标和 before/after；问题 mask 为 `*-ear-accessory-changes.png`。当前记录的 failed contract 没有因 union 域外通过被撤销。

## 归因边界

- previous B 到新方案改变了旧混色边上的 backing 来源和补全域，而前景/alpha/几何固定。
- observed-only 到 virtual-context 同时改变上下文 RGB 和神经推理 mask：前者 `unknown|target`，后者 `dilation(target,3)`。这是联合 conditioning 对照，不能把改善只归给反射 RGB 本身。
- virtual-context 从已允许的原始头发供体反射/近邻延展得到隐蔽背景假设；新增纹理不是原图被遮住区域的真值。
- 审计只覆盖一张图、一个方向；没有 ±30、−100、多原照或性能通过结论。

父任务已收到 37/38 像素失败证据，将在保留本轮结果的前提下生成显式排除软耳/饰品的新 target。新版必须比较 previous B 的新增变化，继承差异应单列；不能把参照改成 initial 后声称全链路对源图耳部完全不变。
