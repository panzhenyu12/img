# 唇部亮度 V2 独立审查

日期：2026-10-15。最终结论：**可接入最小候选 C，作为保持暗纹/色度、修复饱和彩色材质负向失效的限定修复；不代表完整唇部亮度功能已达到 8 分。** 强负向仍暴露原有口缝保护/唇部蒙版边界问题。合同是提亮/压暗唇材质，保持暗纹、口缝、颜色和合理高光；不是单纯让画面像素发生变化。

## 最终 C 的独立结果

新构建已使用独立 Cargo target，冻结身份如下：

- baseline binary：`d0fe0b5251f2f4e862ee20dc7381497fde0f198369f7f37b01eb012c8b68a434`。
- C binary：`adc18a9e67414e1aac6e554afe9bbb14813229ed9a37f5bd41670b2914cd364a`。
- C `color.rs`：`7d7bf5594cf3fb56f0e9b9b406bf9b228562e2db5132192592f5147105dfc597`。

独立脚本 `experiments/lip-lighting-v2-20261015/independent-review/verify_replay.py` 已核实实际磁盘二进制哈希不同且匹配报告，以及全部 55 项的 source/graph/output 哈希和实际旧新像素变化计数。五张输入的 0 强度保持原图，纯 lipColor / lipSaturation 对照与旧版逐 RGBA 一致，RGBA alpha 通道保持不变。结果为同目录 `candidate-c-verified-response.json`。

五张原图各检查 −100、−30、0、+30、+100 的原尺寸 contact sheet。正向在用户图和深肤色图明显减少旧版的漂白、材质变浅粉和口缝黑横条；暗纹仍可辨认。强度没有被隐藏成不生效：独立唇部材质域的编码 Rec.709 亮度中位变化如下（0–255 单位，域为 lips polygon 内扣 teeth polygon，仍包含软边界）：

| 原图 | C +100 中位亮度变化 | C −100 中位亮度变化 |
|---|---:|---:|
| user | +16.40 | −20.40 |
| portrait-makeup | +4.50 | −36.63 |
| portrait-woman | +10.20 | −24.61 |
| portrait-hair | +4.14 | −45.03 |
| portrait-dark | +16.06 | −14.20 |

这些域内没有发现与滑杆方向相反的亮度变化。C 的负向已使先前锁死的饱合彩色域实际压暗；正向 max 通道为 255 的饱和彩色像素仍保持原值，这是当前选择保持线性色度后的色域余量限制。

作者另做了匹配平均线性亮度的旧版对照，报告在 `lip-lighting-v2-20261015/matched-brightness.json`。正向旧版较小强度可匹配 C +100 的平均亮度，并继续显示更大的色度改变；这与观察到的材质保持方向一致。该比较的指标由作者计算，本次独立脚本未重新运行搜索。负向用户图、深肤色图超出旧 −100 的暗化范围，报告已注明不能当作等亮匹配。

## 尚未达到完整质量门槛的边界

portrait-hair / portrait-makeup 的 −100 仍呈现明显内唇亮边、外唇材质与周围肤色之间过硬的边界；旧版同类问题也存在。原尺寸带轮廓定位的对照保存在 `independent-review/portrait-{hair,makeup}-minus100-mask-native-1x.png`。青线是 lips polygon，黄线是 inner/teeth polygon，图像未缩小。

该现象不能通过降低默认强度或将镜面高光全部涂暗来掩盖。需要继续核查真实唇材质覆盖、内口缝/牙齿保护与软边缘的关系；当前几何 polygon 和 hard exclusion 没有提供精确材质边界。C 只改变颜色映射，未宣称解决这一分层问题。仍需多种嘴型/唇妆、透明输入、真实浏览器 WASM 与交互测试，完整功能验收继续保持未完成。

本次无新增关键正确性退化阻止最小颜色补丁合入；root 将独立执行共享 Rust 和浏览器 WASM 验证。以下保留失败版本的全过程，避免把失效对照当成改进证据。

## 初次回放的证据错误

初次 `doc/qa/lip-lighting-v2-20261015/report.json` 声称 baseline/candidate-a 两种二进制，但两者实际 SHA-256 均为 `5750af9d85562603f89697d9aaad4fcb3eac186f5012a41423bdef015b8b049b`。全部 55 项输出哈希相同，`candidate_vs_baseline_pixels` 全部为 0。五张 contact sheet 的两列实际上运行相同产物，不能用于声称新算法优于旧算法。

独立审查已原样保存报告和实测哈希到 `experiments/lip-lighting-v2-20261015/independent-review/initial-same-binary-report.json`、`initial-evidence-failure.json`。作者确认共享 Cargo target 复用产物，正在用独立 target 重新构建。测试二进制通过不证明对应 `image-render` 已更新；后续必须核对公开渲染入口的真实二进制与输出。

## 饱和彩色唇材质不能变暗

候选原公式在公共线性光倍率上使用 `scale = gain / (1 + max_channel * (gain - 1))`。当任一通道线性值为 1 时，正负方向的 scale 都恒为 1。例如 `[255,50,70]` 饱和红唇在 −100 时仍不能压暗。这不等同于合理保留白色高光，因为鲜红/粉色材质同样被锁死。

独立 target 重建后，作者保存的公开 CLI 反例 `evidence/cli-probes/isolated-baseline-b-probe.json` 已复现：B 在 −100 下 `[255,50,70]`、`[255,150,180]` 均保持原值；旧算法对应为 `[225,20,40]`、`[199,94,124]`。同时，黑色 +100 由旧 `[82,82,82]` 改成 B `[0,0,0]`，证实 B 的暗纹方向确有改进，但饱和色负向仍须修正。这两项需要同时满足，不能只用黑色保留的通过结果替代整个亮度功能验收。

该情况存在于原照。使用缓存 lips polygon 内部并扣除 teeth polygon，统计 `maxRGB=255` 且最大最小通道差 `>30`：

| 原照 | 唇部统计域像素 | 饱和彩色像素 |
|---|---:|---:|
| portrait-makeup | 4,045 | 134 |
| portrait-hair | 3,532 | 135 |
| portrait-woman | 2,147 | 4 |
| user | 3,295 | 0 |
| portrait-dark | 2,315 | 0 |

例如 portrait-makeup `(470,445)=[255,198,217]`，portrait-hair `(447,465)=[255,166,147]`。域统计只证明存在该输入材料，不代替软 mask 的精确覆盖或视觉质量测量。数据保存在 `independent-review/saturated-material-inputs.json`。

作者已在公开入口记录失败回归，并于 C 将负向改为公共线性衰减，对中性镜面高光单独保护；正向保持色度时，已满通道的彩色像素本来就没有额外 RGB 色域余量，应明确该限制，不能以漂白换强度。C 的新证据及剩余边界见上文。
