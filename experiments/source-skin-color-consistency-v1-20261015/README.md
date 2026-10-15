# 固定 alpha/几何的源皮肤 F/B 一致性实验

2026-10-15。**未验收，不接生产。** 最终候选为 `outputs-v3-final-b/`；已有 V37、B-only、v1、v2 对照全部保留。这个实验修复了一个真实的背景参数不一致 bug，但没有解决整条脸部轮廓的视觉缺陷，不能据此给瘦脸评 8 分。

## 固定条件与目标

输入是用户 1024×1024 原图 `test/results/user-face-contour-20261014/1024/canonical-source.png`，SHA `5f15e92b14c44a468d924af5ac985a7cc7dc1c87d8ab235f2c94b2e88b173c73`。图片含 AI 创作标记，未验证为实拍照片，不能代表完整真人测试集。

继承 V37 的原始 physical alpha、inverse map、形变几何、±100 强度及五官/耳饰冻结。有限背景支持仅采用 `../joint-skin-scene-support-v1-20261015/outputs-v2-consistent/scene-support.npz`，SHA `80068202929bc5ab75435eb17563a9a87a1e5934732eeade62c5754693aad4a6`。不扩大皮肤或背景域，不降低强度，不修改 alpha，不调用 LaMa。

B-only 已证明旧背景颜色估计会将耳饰亮色带入移出的脸沿。修正 B 后，宽银灰条消失，但源软边仍有细线。此次只检验 `C = A F + (1-A)B` 中 F 颜色估计的不一致是否是细线的一个原因。

## 代码模块和合同

| 模块 | 可复用入口 | 输入与输出 | 限制 |
|---|---|---|---|
| 已观测源背景先验 | `../hair-skin-decomposition-v1-20261015/decompose.py::local_prior` | 源 RGB、可信 scene donor mask、固定半径 → prior、distance | 原 donor RGB 精确保留；未知位置用 donor 有效 5×5 均值与最近点延拓。隐藏 B 是先验，不是真值 |
| 同侧 observed scene 支持 | `../joint-skin-scene-support-v1-20261015/run.py::color_connected` | 独立 full/crop skin 排除、耳饰/耳朵/头发/五官 veto、真实 scene seed、源 Lab → 连通支持 | 只在既有两个 context 内；跨材质、孤立封闭孔洞及域外不得当背景 |
| 有限域 B 替换 | `../alpha-feasibility-skin-v1-20261015/replace_v37_background_feather.py` | V37 trace、finite prior、既有冻结 → 最终 `background-replaced.npy` | 域内 8px smoothstep 只作用 B，不能消灭有限域下沿接缝；脚本是实验调用方，不是生产 API |
| 局部 F 求解 | `solver.py::solve` | 同坐标 C/A/F/**最终 B**、observed opaque skin core、allowed → F 与拒绝/裁剪诊断 | 不估计 alpha 或材质；仅 alpha≥.1，邻域存在可信皮肤 donor 才反解 |
| 字节级颜色流程 | `pipeline.py::correct_foreground` | 上述输入 + 固定颜色过渡 → 实际采样用的 F bytes | v3 只允许最终 B 一个参数，避免反解和合成传不同 B；此测试集量化后也无残差增大，不是任意输入的量化单调性证明 |
| 原生采样/合成重放 | `compositor.py::sample_f32`、`compose` | 原生 inverse、A、F bytes、最终 B → premult samples 与输出 | 明确保存 Rust f32 运算次序；不负责几何或图层正确性 |

scene 的两个 context 为 `[306,389,417,532]`、`[600,394,712,535]`，共 31,665 个 finite 像素。外部 `sceneDistance=inf`；调用方必须检查 finite，不能使用域外 prior。颜色反解、拒绝和最终合成必须使用**同一个最终 B**，包括已有的边界过渡。

源颜色域沿用 v2：raw A∈[.1,.995)，direct full skin>.05，ear/hair/features<.05，jewelry<.001，并有半径 6px 的可信 core。core 需要 raw A≥.995、full skin≥.995、相同保护门槛。6px 为三格原 parser 网格。

反解 `F*=(C-(1-A)B)/A`，用局部真实源皮肤各 RGB 通道的 min/max±1 byte 约束；这是 RGB 盒，并不是颜色凸包或材质真值。A=.1..2 的 smoothstep 仅降低颜色纠正权重。候选若令任一 RGB 源残差增大，整个像素拒绝纠正。固定 finite 域的 8px 内过渡再次作用于 F 颜色后量化；alpha 不变。

原生残差合成式不变：

```text
Cout = C + sample(F * A) - F * A + (A - targetA) * B
```

只有原有 changed ∩ finite scene ∩ !frozen ∩ !opaque 可以写最终图。源 F 颜色修改经完全相同的 inverse 运送；原始 F 的重放 sample 与 target alpha 对 native trace 差异均为 0。

## 失败与修正都保留

1. `outputs/`：初版直接投影局部颜色盒。右侧 `(640,470)` 原 F 比盒约束更接近源观测，投影反而压暗输出；拒绝该版。
2. `outputs-v2-monotonic/`：加入源残差单调拒绝，445 个候选中纠正 218，拒绝 207。初看单元测试通过，独立集成审计发现它用 pure scene B 反解，实际合成用过渡后的 final B；13 个像素对实际 B 的任一通道残差增加，最大 4.09 byte。保留 `qa/post-quantization-source-fit.json`、`qa/pipeline-v2-mismatched-b.py` 及 `qa/final-b-red.log`。
3. `outputs-v3-final-b/`：只修复这个实现不一致。最终 B 用于反解、单调检查和合成。没有调整门槛、邻域、alpha、背景域或几何。

| 指标 | v2 +100 | v3 +100 | v2 -100 | v3 -100 |
|---|---:|---:|---:|---:|
| 源 F 候选域 | 445 | 445 | 445 | 445 |
| 量化后源 F 修改 | 218 | 233 | 218 | 233 |
| 投影单调拒绝 | 207 | 193 | 207 | 193 |
| 相对 B-only 成图修改 | 426 | 445 | 406 | 428 |
| 相对 v2 成图修改 | — | 46 | — | 49 |

v3 两种强度在**实际 final B** 下量化后的源残差增大像素均为 0。0 输出与原图逐字节一致；它通过同一流程验证抵消，而非用原图覆盖后宣称通过。所有强度的域外写入、frozen 改动、不透明耳饰改动、输出 alpha 改动均为 0。源 alpha、inverse、geometry 的 SHA 保持一致。

0/±100 所读取的 final B 数组按旧 trace 的 write/frozen 流程保存，不能据此认为已获得跨强度统一的物理 F/B 分解。未来统一图层模型需要从源图一次建立明确的 B 与未知域；本实验是固定 trace 的因果验证。

## 测试、原尺寸验收与结论

`test_solver.py` 有 6 项测试，包括已知纹理软边、源皮肤纹理保留、小 alpha 拒绝、未知材质拒绝、无 donor 拒绝、错误背景不能强迫颜色出界，以及局部颜色盒不能损害原本更好的源拟合。`test_final_background.py` 的 2 项真实集成回归保留 v2 的 13 像素反例并要求最终 B 一致。8 项全通过，日志 `qa/final-b-green.log`。

已知 A/F/B、固定 11px 平移的合成纹理样例中，最大误差为 V37-like 181.05 byte → B-only 11.48 → F+B 约 2.8e-14。只能证明条件成立时残差细线可由 F/B 一致性消除，不能证明实图 A/B 真实。

独立 `verify_saved.py` 不调用 runner，不重新出候选；从保存 PNG/NPZ 重算源残差与写入域，并核对源、alpha、geometry、inverse、最终 B、输出 SHA。0/±100 全通过。真正 1× 全脸三列图在 `qa/final-verification/{100,-100,0}-face-native-1x.png`，每列 443×512，没有缩小。两侧细节在 `outputs-v3-final-b/{100,-100}/{left,right}-1x.png`。

看过以上全部原尺寸面板后的结论：宽灰条改善主要来自此前 B-only；v3 相对 v2 只有局部细微差异。**旧轮廓细线仍在，背景先验有限域的下沿接缝仍在，整体脸/颈曲线仍不自然。** 因此数值回归通过与视觉验收分开记录，本实验保持 `accepted:false`，不接生产，不再沿旧 trace 扫阈值。

可直接复核：

```sh
services/background-worker/.venv/bin/python -m unittest discover -s experiments/source-skin-color-consistency-v1-20261015 -p 'test_*.py' -v
services/background-worker/.venv/bin/python experiments/source-skin-color-consistency-v1-20261015/verify_saved.py
```

`run.py` 保存 v2，`run_v3.py` 保存唯一 v3；二者都拒绝覆盖已有输出。v2 报告 solver SHA 对应 `qa/solver-v2-with-unused-nan.py`；当前 solver 只增加域外 infinite sentinel 的数值守卫，输出等价已记录在 `qa/nan-guard-no-output-change.json`。失败候选及红灯日志不删除。
