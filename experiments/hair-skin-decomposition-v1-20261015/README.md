# 发丝/皮肤软边分解与遮挡次序实验（2026-10-15）

本目录是隔离的算法诊断，**没有接入生产，没有部署，不是 8 分验收结果**。冻结 V36 几何、原图、选人范围和原始解析，验证两个不同问题：旧轮廓混色怎样拆分；固定发丝纹理是否应始终保持最终可见。它没有自动推断深度次序，也没有修复耳饰。

## 当前证据与结论

最新结果为 `occlusion-corrected-prompts/`，包含三张原图的 ±100，以及 woman +100 的完整 LaMa 128/256/512 backing 组合，共 9 组、18 张结果。所有 A/B 的几何、颜色求解和输入一致；没有按图挑选一个 A/B 作为最终算法。

| 原图/方向 | 观察 | 限制 |
| --- | --- | --- |
| woman −100 扩宽 | A 保留脸侧断续黑线；B 显著消除该黑线，并形成连续外轮廓。B 的新覆盖连通区 12861 像素，其中遮挡原 hair core 4284 像素。 | B 是“皮肤在发丝前”假设，不能据此宣称自动次序正确；局部曲线仍需验收。 |
| woman +100 内收 | B 去掉大部分旧亮边。A 出现浅色残条、阶梯边，失败。 | B 仍有暗平背景条和细轮廓边。完整 128/256/512 backing 均未解决，全部保留。 |
| hair −100 扩宽 | B 减少耳旁黑色突刺；A 保留突刺。B 新覆盖连通区 2583 像素，其中遮挡 hair core 1077 像素。 | 不能把前后顺序假设当通用自动修复；脸侧/颈侧轮廓仍有缺陷。 |
| hair +100 内收 | A 在脸侧增加细碎残边；B 改动较温和。 | 背景条带、轮廓硬边未根治。 |
| user ±100 | A/B 都没有新写入，结果与冻结 V36 字节相同。 | 当前头发边域位于基本不动的头顶，用户图的耳饰缺陷不在本实验范围，不能声称解决。 |

优先看原始分辨率对照：

- `occlusion-corrected-prompts/portrait-woman--100/v36-backing/left-A-B-1x.png`
- `occlusion-corrected-prompts/portrait-woman--100/v36-backing/right-A-B-1x.png`
- `occlusion-corrected-prompts/portrait-woman-100/lama-256/left-A-B-1x.png`
- 对应 `lama-128`、`lama-512`、`v36-backing` 保留全部对照，不能只看较顺眼的一张。
- `occlusion-corrected-prompts/portrait-hair--100/v36-backing/face-A-B-1x.png`

结论是：**不拉伸头发纹理，与不允许新的皮肤覆盖原头发，是两种不同约束。** B 的实验效果支持最终可见性冻结是扩宽黑线的一个原因。但仅靠原图的 hair/skin alpha 不能确定深度顺序；后续需要显式次序输入或独立可靠的次序推断，不能默认把所有 hair 放到皮肤后面。

## 输入与 ZIM 接口纠正

旧 ZIM 调用同时提供 boxes 和点时，缓存的官方 decoder 会覆盖 point coordinates/labels。旧“正负点无效”的解释不成立：负标签没有真正送入 decoder。旧结果完整保留在 `experiments/hair-accessory-matting-zim-v2-20261015/`，其 README 已修正。

本次最新分解使用 `experiments/zim-combined-prompts-20261015/outputs/<photo>/component-box-positive-skin-negative-1-combined-box/alphas.npy`，是真实的 box + signed point 混合编码；三张图都用官方 score argmax 2。四个 float alpha 和 prompt 编码证据由上游完整保留。**没有使用事后手选更好看的候选。** 历史 `source-decomposition/` 则使用旧语义选择结果，不能与 corrected-prompts 混称。

来源与哈希：

- `source-decomposition-corrected-prompts/report.json`：原图、graph、原 alpha、solver、上游 corrected prompt report 的 SHA256；9 个颜色变体与原尺寸 crop。
- `occlusion-corrected-prompts/report.json`：颜色层、逆映射、脚本 SHA256、写入与遮挡计数。
- `occlusion-corrected-prompts/validation.json`：18 张输出及完整 initial backing candidate 的 SHA256，以及逐图保护不变量复核。
- 冻结几何来自 `experiments/face-material-fill-v1-20261015/{woman,portrait-hair,user-face-contour}-{100,-100}-full/`；本实验没有重新拟合轮廓。

## 软边分解

`decompose.py` 对原图混色 `C = a H + (1-a) S` 做局部稀疏二次求解：H 为头发颜色、S 为皮肤颜色、a 为 learned hair alpha，带独立原图 donor 先验和相邻层颜色正则。先用独立 donor 对 alpha 做有界修正，再求颜色；允许变化最多 ±0.15。先拟合颜色再反推 alpha 的失败变体已保留，避免“重建源图很好”掩盖隐藏颜色错误。

仅处理两种可靠 donor 均在限定半径内的原有混色域。不是扩张整个人脸蒙版，不会把所有 hair mask 统一改成 skin。source reconstruction 最大通道误差 ≤4/255 只是拒绝门槛，**不是隐藏颜色真实的证明**。当前 sRGB 版本在这些 JPEG/PNG 样本上重建误差较小，不意味着真实成像一定在 sRGB 中线性混合。

| corrected source-fit-srgb | 求解域 | 通过重建门槛 | 全求解域平均 RGB 误差（byte） |
| --- | ---: | ---: | ---: |
| woman | 11141 | 10832 | 0.438 |
| user | 426 | 426 | 0.129 |
| hair | 1941 | 1869 | 0.503 |

woman 另有 131 个严格原图观测支持的旧 hair 误标回收；未加入 accessory 回收。线性光版本、超出色域/门槛的像素与统计均保留。该求解属于普通带先验的欠定逆问题，**没有新训练模型，不是新 SOTA 声称，也没有解决单张照片隐藏颜色唯一性**。

## 两种明确的遮挡假设

`occlusion_hypotheses.py` 始终输出 `layerOrder=UNKNOWN`。

- **A：头发在前。** 纹理/alpha 固定在原坐标，移动下面的皮肤。原混色边的底层 skin coverage 从 visible physical coverage 除以 `(1-hair alpha)` 得到。opaque hair core 最终 RGB 与原图完全相同。这一底层延拓在内收例子中出现浅色残条，属于失败，不能当作正确重建。
- **B：皮肤在前。** 移动 premultiplied 皮肤，原头发作静止 backing。只有 `transported skin alpha > source skin alpha + .001` 且处于原 hair≥.05、选中人≥.5、ROI、非耳饰区的新覆盖，才允许遮住原 hair core。新覆盖连通域还必须接触原观测 mixed/pure 边界的一像素邻接；邻接操作只判连接，不扩大写入域。未连接的覆盖留为 unknown。

两种情况都没有重采样 hair 纹理坐标。B 仅改变新皮肤下面的最终可见性，未覆盖的 deep/far hair core 字节不变。所有新写入排除 ears/accessories≥.05；选区外与 backing 初始候选相比不变。源 mixed band 与 transported mixed band 也会进行颜色去混，这是与纯新覆盖计数不同的边缘修正域。

### 完整 backing 组合

woman +100 的 LaMa 变体以 backing agent 的**完整** `candidate.png` 为 initial，然后施加本实验限定边缘修正，并非只保留两者 mask 的交集。所有完整 backing 输出来源：`experiments/face-background-context-20261015/woman-100-local/{128,256,512}/`。旧混色边原 RGB 不能继续被当成隐藏纯 hair backing，因此该源混色域被 H 估计替换。该颜色估计仍不等于可靠隐藏纹理。

## 验证与失败记录

`test_synthetic.py` 使用已知真实 alpha/颜色的独立合成场景。5 个固定/纹理、linear/sRGB、准确/偏差 alpha 场景通过：纹理真实 alpha 的换 backing 平均误差 0.686 byte；偏差 alpha 的线性和 sRGB 版本分别 0.461/0.405 byte。常色精确例子接近机器误差，opaque hair 固定不变。原生 bilinear inverse sampler 与 Python replay 的最大 RGB 差约 `3.81e-5` byte，alpha 差约 `1.18e-7`，已有 `renders/report.json` 记录。

`test_occlusion.py` 验证真正新增且接触源边的新覆盖可以进入，孤立远处块保持 unknown，保护带阻断连通传播，已有覆盖/非 hair 不得进入，transported-edge 写入路径不能绕过 unknown 拒绝。另独立读取 18 张 A/B 结果，验证所有输入 SHA、允许域外不变、耳饰无新增改动、未覆盖 hair core 不变、user 两向零改动。**这些都是实现约束测试，不是审美 8 分测试。**

保留的失败记录：

- `initial-fit-before-alpha/`：源图重建误差小，但换 backing 平均误差约 11.87 byte。
- `alpha-prior025-alternating-failure/`：换 backing 仍约 3.88 byte，超过原 2 byte 门槛。
- `synthetic-initial-fixture-core-tolerance-failure/`：合成 exact fixture 将 alpha≥.999 当成纯色引起误差；改成物理纯 0/1 donor，原算法及门槛没降。
- `initial-source-anchor-assumption-failure/`：woman 模型最大 alpha 约 .994，旧 .995 donor 假设无锚点；改为同时要求原语义核心/内距和 alpha≥.95，并完整记录。
- `corrected-prompts-initial-output-path-failure/`：相对输出目录错误，修正 `.resolve()`。
- `occlusion-inherited-protection-assertion-failure/`：首轮错误地将旧 V36 已改动的软耳饰边与原图比较。新写入一直排除≥.05；正确新不变量是相对完整 initial 不再改动，并单列 inherited source 差异，未放宽写入域。woman ±100 inherited soft protected 差异为 42/54 像素，user 为 230/292 像素，hair 为 0。
- `occlusion-before-unknown-write-gate/`：独立审计用合成反例发现，断开的新覆盖可以经 transported-edge 写入绕过 unknown。`independent-occlusion-audit/test_unknown_gate.py` 保留旧脚本反例（6 个越域像素）；当前 `write_domains` 对两种写入路径显式排除 unknown。真实 9 组 unknown 均为 0，修补后图像应保持逐字节相同，另有复核记录。

## 复现

使用 `services/background-worker/.venv/bin/python`。无需下载新模型，ZIM 原始输出和 V36 原尺寸 layers 必须已存在。

```sh
services/background-worker/.venv/bin/python experiments/hair-skin-decomposition-v1-20261015/test_occlusion.py
```

生成脚本默认对已有输出目录拒绝覆盖，防止覆盖历史证据。`run_real.py --corrected-prompts --out <新目录>` 可生成新 source-only 分解；`compose_real.py --estimates <report> --full-backing-combination --out <新目录>` 是旧 skin-front/核心冻结诊断（不等同于新 A/B）。`occlusion_hypotheses.py` 本次冻结输出位置固定，重跑须先保留旧目录；不要覆盖后继续沿用旧 SHA 报告。

下一项实质工作应是验证可靠遮挡次序来源及旧轮廓后方纹理，不应继续盲目调颜色阈值来掩盖层级问题。B 不作为生产默认，A/B 失败都纳入后续验收。
