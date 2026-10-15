# 独立审计：固定发丝坐标与固定发丝可见性

2026-10-15。结论：A/B 遮挡实验有诊断价值，仍不通过完整瘦脸画质验收，不支持 8/10 或上线结论。生产代码没有由本审计修改。

## 审计对象与证据

- 三张冻结原照，正负 100；woman 正 100 加 LaMa 128/256/512，9 组、18 张输出。
- 几何来自冻结 V36 inverse map；不重新检测、不换图、不调滑杆强度。
- 新实验确实读 `source-decomposition-corrected-prompts`，后者取修正 ZIM signed-point+box 调用的官方最高分 index 2。不是拿旧 alpha 充作新调用效果。
- 独立像素、SHA、mask 检查见 `audit_outputs.py` / `report.json`。未修改生产源文件或旧 Rust 回归。
- 原合成脚本 SHA `f8c51cc8342f7691ac9a481cae9fbd05bd340697b49f0f5953e92be684476c7b` 已冻结到 `frozen/`；unknown 修复后 SHA `8ba4498917515b9c719b486d65e18e0464860129c01520968067e25357140de9`。

## 遮挡因果判断

发丝的纹理坐标固定，不代表其输出可见性也必须固定。若头发位于脸后，扩脸可以遮住原来可见的头发，头发纹理没有移动；若发丝位于脸前，皮肤移动应保留前景发丝。单张 alpha、语义类别和连通性不能确定两者的深度先后。

A 将 hair_core 的输出像素固定为源图。B 保留未被覆盖的核心像素，仅允许与源混合边界相邻的新增皮肤覆盖头发。两者均未用脸部 inverse map 采样 hair_color，移动的是皮肤层。B 中的发丝可见性变化不能称作“头发拉伸”。

| 原照/强度 | B 允许覆盖的头发核心像素 | B 实际改变的核心像素 | A/B 在这些核心的平均 RGB 绝对差 | 独立观测 |
| --- | ---: | ---: | ---: | --- |
| woman −100（扩脸） | 4,284 | 3,957 | 79.94/255 | B 更连续地覆盖原地黑边，A 保留了旧脸旁的深色轮廓 |
| hair −100（扩脸） | 1,077 | 1,018 | 56.87/255 | B 去掉图像右侧脸/耳旁大部分黑条，仍有耳侧残留与过度鼓起的轮廓 |
| woman/hair +100（收脸） | 0 | 0 | 不适用 | 暴露区域重建和旧混色边缘仍未解决 |
| user ±100 | 0 | 0 | 不适用 | A/B 与初始 V36 解码像素完全相同，无改善 |

独立用 SciPy 采样物理皮肤 alpha，发现 woman 扩脸有 1,165 个、hair 扩脸有 249 个头发核心位置已被 alpha>0.95 的移动皮肤覆盖，但 A 仍显示原头发。这支持“固定输出可见性造成部分黑条”的因果解释。几何映射在 A/B 中相同，故不能把这些差异归因于新的几何连续性改善。

不过 A/B 也采用不同的皮肤层解释（under_skin 与 visible physical skin），不是只开关一次输出冻结的单变量实验。B 能覆盖黑条并不证明头发在真实场景中都位于脸后；错误的 B 假设可能抹去真实前景发丝。它也不能反证其他位置不存在 alpha 或几何问题。

## 原分辨率视觉复核

- woman +100：A 左脸明显出现重复浅色边、断续硬边，拒绝。B 减轻原亮边，但 V36 backing 留下宽而平滑的棕色带。LaMa 256 的 B 改善部分露出纹理，却仍有偏硬黑边、耳侧突刺；原 V36 嘴角局部变形也仍在。
- woman −100：B 外脸边比 A 更连贯，但这只是脸在头发前的假设结果，不能据此选择整张图的统一前后顺序。
- hair +100：A/B 均有脸侧断续锯齿/硬边和下颌至颈部不自然连接。B 没有达到可用精修质量。
- hair −100：B 明显减少原 V36/A 的黑条，但耳侧仍有小残留，脸侧曲线在极值下仍不自然。
- user ±100：均为 no-op；原有耳饰旁黑楔、局部轮廓异常未被解决。此实验不能计作用户原图修复。

相关原图尺寸面板均在 `../occlusion-corrected-prompts/`，已检查三原照的正负极值，以及 woman 左右脸和 LaMa 对照。未使用缩小图替代原尺寸判断。

## 禁写域与新增发现

18 张图独立验证：写域外、ROI 外、person coverage<0.5、未移动位置和耳朵/饰品语义覆盖≥0.05 均无新增改动；输出 alpha 保持源图；A 的头发核心与源图相同，B 未被覆盖的头发核心与源图相同。

“无新增改动”的参照是该组 initial（V36 或 LaMa candidate），不能改写为“原图保护区域完全没变”。V36 继承的耳朵/饰品差异仍有 woman +100/−100 的 42/54 像素，user +100/−100 的 230/292 像素。先前失败的源图完全一致断言及旧回归没有被删除；新增合成步骤的合同比较不能替代全链路合同。

当前 `allowed` 不直接消费通用 protectMask、用户 selection mask 或独立的人物身份。user graph 包含 `mask`，但 user 案例本次写入为零；其通过不能证明有选择范围时的写入约束已经实现。person coverage 和 ROI 也不能自动等价于其他人物保护。

独立发现的真实逻辑漏洞：`unknown` 只标记不接源边界的新增覆盖组件，旧 `writes` 的 `edge` 又包含 transported_border，因此可能从第二条通路写入 unknown。`test_unknown_gate.py` 的有限平移反例让 6 个 unknown 像素全部被旧逻辑写入。真实 9 组 unknown=0，原有测试无法暴露它。

作者随后引入 `write_domains`，两支统一排除 `unknown_a | unknown_b`。独立反例验证新逻辑 A/B 都为 0；前后 90 个 PNG 文件逐字节一致。修补仅封住合同缺口，不是画质提升或遮挡深度推断。完整红/绿证据在 `unknown-counterexample.json`，失败实现保留于 `frozen/` 和作者的 `occlusion-before-unknown-write-gate/`。

## ZIM 归因审计

已核对 vendor 的实际调用：同时给 points 和 boxes 时，decoder 的 boxes 分支覆盖 point tokens 为框角和 [2,3]，predictor 仍根据所有点坐标构造 attention，且不区分正负标签。旧 mixed 结果不能表述为“模型收到负点但忽略了标签”。

新 adapter 显式拼接 signed points 与 box corners，point-attention / box-attention 独立控制，单提示委托给未修改 vendor。3 项 feed 合同测试通过；6 组 upstream-mixed alpha 与历史数组独立检查逐浮点相同。最早 `zim-20261014/run.py` 的 box/points 互斥实验不受本次修正影响。

以下比较不可混用：

1. 同一新 A/B 对固定同一 inverse map、同一 corrected alpha/颜色推断、同一 backing，比较的是两种层级解释。
2. V36 对新 A/B 同时改变来源 alpha、前景/背景颜色分解和合成，不能把全部改进只归给遮挡规则。
3. 旧 `renders/` 对新 `occlusion-corrected-prompts/` 还改变了 ZIM prompt 输入，不可倒推撤销旧实验的实测失败。
4. source-fit-srgb 的低重建误差只说明能拟合原混色 RGB，不能证明两层颜色、物理 alpha 或深度顺序真实。women 最大通道重建误差仍约 34/255，且 alpha 被允许调整至 0.15。

后续应将源图混色边缘、后层暴露纹理、前后遮挡关系以及耳饰附着几何分别建立可失败的对照，再做联合候选。当前 A/B 保留作诊断与研究依据，完整瘦脸候选继续拒绝。

## 复现

```sh
services/background-worker/.venv/bin/python experiments/hair-skin-decomposition-v1-20261015/independent-occlusion-audit/audit_outputs.py
services/background-worker/.venv/bin/python experiments/hair-skin-decomposition-v1-20261015/independent-occlusion-audit/test_unknown_gate.py
services/background-worker/.venv/bin/python experiments/zim-combined-prompts-20261015/test_adapter.py
```
