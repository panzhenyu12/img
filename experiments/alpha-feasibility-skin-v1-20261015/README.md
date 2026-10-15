# 有条件的皮肤 alpha 色域可行性实验

只处理**已确认 skin-over-scene 材料、且背景 RGB 可视为已知**的局部反解。它不识别材料、不估计隐藏背景、不解决未知物理 alpha 的信息不足问题。未改 Rust、WASM、服务或原候选。

对于 `C=aF+(1-a)B`、RGB 各通道在 `[0,1]`：

`a_min = max_c(0, (C_c-B_c)/(1-B_c), (B_c-C_c)/B_c)`。

分母为 0 的非正分支取 0。将 alpha prior 投影到 `[a_min,1]`，这是满足色域约束的最小改动；不是对 alpha 真值的恢复保证。随后反解 `F=B+(C-B)/a`。未确认材料的 alpha/RGB 保持逐值不变。

## TDD 与已知真值实验

`qa/red.log` 保留未投影版本 5 tests 中 3 fail 的证据，`qa/solver-red.py` 保留该实现。加入可行性投影后 `qa/green.log` 5 tests 全过：已知背景高光、非饱和可行 prior 不动、黑/白背景和零 alpha、随机 1024 个真实分层像素、未确认发丝/金属/背景保持不变。

- 真高光皮肤的 alpha 仅低估 `.02`，原 clipped 反解后旧轮廓残线最大 **4.9215 byte**；投影后最大误差约 `5.7e-14 byte`。
- 修正后的有效 no-clipping 对照：只在 alpha `.35..65` 修改 `.02`，无 clipping/vanishing-alpha fallback，常量已知背景下输出误差 `2.8e-14`，prior 未变。这个对照说明错误 alpha 并非必然产生残线。
- **不可辨识反例**：同样可行但不是真值的 alpha，源重建误差 `2.8e-14`，空间变化背景下的移动结果误差仍为 **2.1165 byte**。色域可行与源重建完美均不是 alpha 真值。
- **错误暗背景反例**：真实 scene 像素 `(17,10,2)`，邻近背景估计 `(19,15,6)`，仅 5 byte 差就会把 prior `.0021` 推到 `.6667`。必须拒绝未知背景/材料，不能全域应用。

图像、数值及 SHA 在 `qa/report.json`、`qa/known-highlight*.png`、`qa/nonidentifiable-spatial-scene*.png`。

## 真实原图结果：不推荐接入

对原图先前独立观察的 **235 个脸颊皮肤路径像素**，使用新的 direct skin alpha：

- 可行性违反 **0**；alpha 和 foreground 改动均 **0**。
- 没有可视改善，不能称为候选通过。
- 宽域 `local ∩ jewelry<.001` 有 79 个违反，但大量是暗背景/耳饰孔洞像素。例 `(642,475)` prior `.0021 → lower .6667`，`(644,454)` `.0226 → .6667`，`(380,495)` `.0088 → .6`。它们不是确认皮肤，已作为 rejected mask 保留，没有渲染修正。

`qa/real-component-{1,2}*.png`：原图、确认皮肤路径（绿）、拒绝扩大区域（粉）。`qa/real-source-feasibility.npz` 保留全部数值。

**决策：已知背景的数学子问题成立，但当前照片没有可信适用域带来改善，不能作为通用修复接入 Rust。** 需要先解决 source ownership、背景可信度，以及不同 alpha 层交界的合成合同。

## 原 delta audit 的解释边界

`experiments/joint-skin-matting-v1-20261015/test_delta.py` 保留所有原失败结果。其残差公式精确等于 `C_target_model + (C_source-C_source_model)`；保留的是源坐标的固定残差。9 tests 中 5 fail 是未知输入误差下的质量反例，并非宣称无信息的 compositor 有义务恢复真值。

其中原名为 no-clipping 的控制实际包含很低 alpha 的 clipping/fallback，因此不是有效 no-clipping 控制。本目录显式修正了对照条件，没有改原失败脚本或结果。

## 运行

```sh
services/background-worker/.venv/bin/python experiments/alpha-feasibility-skin-v1-20261015/test_solver.py
services/background-worker/.venv/bin/python experiments/alpha-feasibility-skin-v1-20261015/audit.py
```

## 后续 V37 原生追踪

`trace_v37.py` 对 V37 隔离源码仅添加 trace，原源码快照在 `qa/v37-trace/face-before.rs`，独立target构建并冻结新binary。±100 与原 `outputs-delta` PNG逐像素相同，几何hash相同，f32公式重放0差异。

关键结果：`(630,503)` source alpha=.97409，旧混合缺陷已解，残差仅[-1.57,-.04,.70]byte；其+100输出灰[101,95,93]来自B=[103,95,92]。左`(375,450)` source=.99456、target≈0、非frozen、changed=true，残差≤1.44byte，但B=[238,227,208]导致银白条。右`(640,460)`残差≤.066byte，B=[119,120,123]直接成为输出灰色。因此当前这些点的主因是B估计，而非alpha/固定残差。

`qa/v37-trace/{100,-100}/{left,right}-layers-5x.png` 依次为原图、输出、F、B、残差×5。`report.json` 保留14个坐标×2方向的完整数据。未修production，后续应审查背景材料供色与fallback。

## B-only 真实修复实验

`replace_v37_background.py` 仅在审核过的 consistent scene有限域，用same-side可见scenePrior替换V37 B；`replace_v37_background_feather.py` 增加有限域内8px B写入过渡。其余alpha/F/samples/geometry、opaque与frozen保持。输出位于 `qa/v37-scene-b-only-v1`、`qa/v37-scene-b-only-v2-feather`，每组0/±100。

左银白/右灰条显著消失：左(375,450) [238,228,209]→[20,18,14]；右(640,460) [119,120,123]→[31,24,18]。但下端finite域截止造成小楔/接缝，旧边剩几byte残差（右641,470约7.1byte）。因此大B污染因果成立，但整体尚未通过，accepted=false。未扩scene域或改rawalpha/F以掩盖问题。

`verify_b_only.py` 独立核对六组保存输出的圈外/有限域外/金属/frozen/输出alpha不变、0exact、alpha/F/samples/geometry SHA；报告 `qa/v37-b-only-verification.json`。完整评审见 `doc/qa/alpha-feasibility-skin-20261015.md`。
