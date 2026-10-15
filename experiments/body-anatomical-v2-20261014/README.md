# 身体源曲线候选 V2：有限改善，未启用

2026-10-14。保留 V1 的源码、固定二进制和92项输出；V2 独立源码、Cargo target、不可变二进制和完整同图回放。生产身体、公共native/WASM产物、检测和服务均未改动。

## 单变量问题和修正

V1 用正部 `max(r-rK,0)` 表达大腿相对远端膝参考的多余半径。站立左腿一侧 rK=52.94px，而 t=.5/.625/.75 的源半径是50.37/51.15/52.47px，导致大部分该侧被关闭，最大位移仅0.171px。

V2 把偏差改为 `d=abs(r-rK)`，仍用 `g=.85*d²/(d+max(.04*rK,1))`，平滑限制为 `.26*r`。g的源半径导数在−.85到+.85之间，故 `r±g` 都严格递增；源自身的增减和转折顺序得以保留。26%仅是整半径位移上界，并非实际体围减少量。

唯一附带的必要边界修改：V1的正部在膝参考附近天然归零，直接换abs会在站立膝边界留下0.040–0.172px位移残差。因此**只对新增r<rK分支**在t=.80到.875平滑归零，原正差分支不变。源序保证适用于轴干t=.22到.80；固定端过渡、原保护区和缺失截面锚定必须另验，不能将标量函数定理外推到所有成像像素。

等宽/膝宽不可测时仍用V1已验证的连续可信度混合；没有移除原圆柱局部效果测试。对中间可信度不宣称无新极值。没有合入v18 C1纹理延拓。

## 同输入证据

旧5图15部位、独立官方MediaPipe瑜伽原照片8部位，均固定graph和检测输入，在−100/−30/+30/+100回放，共92项。所有来源/graph哈希与V1相等，旧基线输出逐字节相等；每次调用前后检查固定renderer SHA。对比图保留原尺寸，不用重检测后的不同mask混比。

| +100部位 | V1每侧最大位移px | V2每侧最大位移px | V2轴干平均位移px | 同平均位移旧参数 |
|---|---:|---:|---:|---:|
| 站立左大腿 | .171 / 8.656 | 1.877 / 8.656 | 3.287 | +34.374 |
| 站立右大腿 | 7.910 / 8.732 | 7.910 / 8.732 | 5.061 | +51.780 |
| 瑜伽左大腿 | 6.548 / 7.361 | 6.548 / 7.361 | 2.440 | +56.345 |
| 瑜伽右大腿 | 8.377 / .989 | 8.377 / .989 | 3.605 | +79.318 |

所有16个大腿项的匹配平均位移误差最大 .007754px。这里的位移由引擎实际逆映射求解，包含选区和保护衰减；它表示被测轮廓的几何目标，并非另一个人工真值分割测得的最终照片边缘。逐截面原半径、目标半径和运动保存在 `renders/*/*-curve.json`。

保护像素、支持域外、纯收缩已知背景/部位外、原照片alpha变化均为0；扩张允许覆盖背景。腰、胯、上臂、小腿共76项与旧版/V1逐字节一致。其余16项为4条真实大腿的四档。汇总见 `evidence/summary.json` 和 `evidence/v1-v2-summary.json`。

## 视觉决定

- 站立右腿：同平均位移的旧算法仍有中下段凹陷，候选延续源渐细。说明有独立于降低总体位移的几何改进。+100曲线与照片均支持此局部结论。
- 站立左腿：abs修正了被正部关闭的一侧，最大运动恢复到1.877px；总平均运动只比V1多约0.193px，两侧效果仍高度不均，不能宣称解决完整瘦腿控制。
- 瑜伽：V2几乎没有改变V1结果。−100主要鼓起短裤，裤脚与皮肤交界仍不够自然；屈膝腿的连接质量也未得到新收益。
- ±30：完整看过两张图、四条腿、正负档原尺寸对照，V1/V2及匹配旧版差异很弱，没有稳定可见的总体自然度优势。
- 原横向分段线性采样、布纹折角风险、源alpha/衣物分区误差及新露出背景填补均未修复；候选无法凭源衣物轮廓推断被覆盖肌肉的解剖目标。

**结论：V1/V2作为有限的源曲线实验保留，不进生产，不认定身体自然度达到8分。**下一步应区分衣物/裸露部位/关节连接，而不是进一步降低所有强度。数学源序和零背景差分不能替代真实画质验收。

## 验证与产物

31个默认body测试通过（`body-test.log`），新增负偏差源序、固定膝接续以及confidence=1的透明RGBA/固定核心/另一人物整图行为测试。没有声称native/WASM已做此候选的专项对齐。

`evidence/*-v1-v2.png`：原图/V1/V2/匹配旧版四列；`renders/*/*-compare.png`：原图/原强度旧/V2/匹配旧；`evidence/*-curves.png`：源及三种实际目标曲线；`evidence/small-strength-*.png`：±30复核汇总。每份完整来源与输出哈希在 `renders/*/report.json`。

`v1-to-v2.patch` 为后继差分，`candidate.patch` 为含独立测试的完整实验差分，均以实验instrumented基线为比较，不能无审查当生产补丁套用。

```sh
CARGO_TARGET_DIR=/Users/modao/work/img/.run/body-anatomical-v2-target cargo test --manifest-path experiments/body-anatomical-v2-20261014/Cargo.toml --release body::tests
services/background-worker/.venv/bin/python experiments/body-anatomical-v2-20261014/replay.py
services/background-worker/.venv/bin/python experiments/body-anatomical-v2-20261014/replay.py --cache test/results/body-anatomical-inputs --samples body-yoga-mediapipe --label yoga-fixed-input
services/background-worker/.venv/bin/python experiments/body-anatomical-v2-20261014/summarize.py
services/background-worker/.venv/bin/python experiments/body-anatomical-v2-20261014/compare_v1_v2.py
```
