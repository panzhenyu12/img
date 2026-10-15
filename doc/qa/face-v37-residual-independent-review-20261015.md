# V37 source residual 合成独立复核（2026-10-15）

**结论：source residual 修正了低 alpha 区域被大幅重绘的合成错误；V37 raw-alpha 整体仍有明显旧轮廓亮线/黑带，不能作为外观验收通过或 ≥8 分证据。** 本复核未改生产代码、未部署；既有 `opacity-bound-red.json` / `opacity-bound-green.json` 原样保留。

## 比较成立的证据

独立脚本 `experiments/face-direct-physical-v37-20261015/audit_independent.py` 实际执行通过，报告及六张原尺寸对比位于同目录 `independent-audit/`。每张依次是 source、V37 raw alpha + 旧合成、V37 raw alpha + source residual；避免把 raw alpha 与 native refined alpha 的差异混在合成器比较里。

核对 3 张原照片、0/±100：原 RGB、ZIM crop-point idx2 raw alpha、graph 与保存的原生二进制 SHA 均匹配报告。两候选 `flow-geometry.json`、`actual-source-width.json` 与对应 `native-v2-*` 的 bytes 完全一致；source-alpha trace 与 raw 模型全图在 ROI 的切片完全一致。0 值逐字节 identity；6 个非零输出的 ROI 外逐字节保留，输出 alpha 全部 255。每组仍有 1.1万–6.9万不透明内部像素发生变化，因此这不是全图原样返回获得的通过。

`debug/source-physical-alpha.png` 写出发生在 `FACE_V37_PHYSICAL_ALPHA` override 之前，不能用于声称实际采样 alpha 一致；本复核使用 override 之后的 `layers/source-alpha-f32le.bin`，并直接比对 raw float。

## 原有 opacity bound 测试是否有效

`test_opacity_bound.py` 检查允许变化的数学上界，不是把生产合成公式再实现一遍。它对应明确前提：输入不透明，颜色范围 0–255，底层场景保持静止，仅改变一层前景的颜色/覆盖率。令 source/destination coverage 为 a0/a1，则变化为 `a1 F1 − a0 F0 + (a0−a1) B`。各系数和为零，正系数之和为 `max(a0,a1)`，因此更紧的逐通道界为 `255 max(a0,a1) + 1 byte`。原测试的 `510 max(a0,a1)+1` 是更保守但有效的界。

原测试只选 `outputs-delta` 自报的 `changed` 域且未断言非空；单独依靠它存在空域/不变图也通过的漏洞。独立复核增加输入/几何/alpha验签、全 ROI 检查、非空域、真实内部改动以及 ROI 外保留，不再以 `changed` 切掉低 alpha 的检查域。target alpha 仍来自 renderer trace；这验证合成合同，**不独立证明模型 alpha 是物理真值**。

| 图片 / 数值 | 旧合成低 alpha 违规像素 | 旧合成最大 RGB 变化 | residual 低 alpha 违规 / 最大变化 | residual 全 ROI 更紧界违规 |
|---|---:|---:|---:|---:|
| user +100 | 3,220 | 237 | 0 / 0 | 0 |
| user −100 | 3,144 | 234 | 0 / 0 | 0 |
| woman +100 | 271 | 19 | 0 / 0 | 0 |
| woman −100 | 138 | 51 | 0 / 0 | 0 |
| hair +100 | 602 | 160 | 0 / 0 | 0 |
| hair −100 | 502 | 160 | 0 / 0 | 0 |

低 alpha 定义为两端均 < .002；这是 bug 的判别域，不是前景/背景真值定义。零字节变化也受量化影响，不能推广为所有亚像素 alpha 都无误差。

## 实际图片仍未过关

逐一看六张 1×对比，而不是只看统计：

- **user +100**：旧合成在耳饰/颈边之外的大块灰色补丁消失，但内收后左下颊旧皮色边仍能辨认。
- **user −100**：大块外部污染消失；图像左侧从耳饰下缘至下颌的黑带/硬连接仍明显。
- **woman +100**：额部和右侧背景更稳定，但图像左侧原脸缘保留一条明显肤色亮线，亮线与新轮廓之间是暗带。两种合成都有，residual 没有解决。
- **woman −100**：外部零散污染有所改善；单图表面更平顺，但不足以抵消 +100 的拒绝项。
- **hair +100**：远处青色背景的错改被消除；右侧旧轮廓细线和左下颊皮色残边仍存在。
- **hair −100**：右耳/下颊邻近的暖色细边与局部连接异常仍在。

合理解释是 source residual 保留了原 RGB 中未由估计前景完整解释的残差。它能防止仅因为背景估计有偏而重绘整片场景，也可能原样保留 alpha/F 解释遗漏的旧脸缘。这里没有隐藏背景真值，不能只由低重投影误差认定具体 layer ownership 正确。

**允许的下一步**：在当前生产核心的独立副本只移植“不透明 joint 情况下 source residual”这一项，保持当前 refined alpha、原图和 graph，重放 0/±30/±100 并核对白边是否倒退。该组件实验不能被 V37 raw alpha 的失败直接否定，也不能被本报告的 bound 通过直接批准。最终需要逐图原尺寸外观比较。
