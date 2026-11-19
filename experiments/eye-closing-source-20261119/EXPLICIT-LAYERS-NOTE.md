# 明确源材料身份后的分层估计

2026-11-19。此候选属于算法实验，不是公开 UI。自动 V6/V7 材料判定失败结论不变；本许可只适用于与源图绑定的显式材料标注。

## 身份未知与模型近似误差的区别

固定粗源笔刷 `[30,16] → [38,20]`、半径 1.5 px，使用源眼裂内明确眼球供体：旧两色 helper 得到 31 个可拟合像素，却将其中 5 个标为 unknown。它们是两个极薄混色边及三个笔刷范围内实际眼球点；不是无供体或非法 RGB。只读测量这 5 个有界估计对最终图像的影响，独立一次运动 oracle 的 RGB RMSE 为 1.369、暗线质心误差 .0517 px，均低于原门槛。

另一 agent 的固定标注 `test_source_material.manual_lash_labels` 更保守，只提供 132 个明确 ocular anchors、27 个 moving 像素。旧 helper 拒绝其中 3 点。三色源模型将颜色拟合误差分别从 4.56→1.41、11.27→4.22、12.70→4.76；最后一点仍超出约 3.71 的经验预测尺度，但有合法源供体和有界可重建源层。

源平面自身存在非线性纹理外推偏差。经验拟合尺度不是材料身份分类器；人工已明确确认同动前景后，不应将每个颜色近似误差再次称作“材料未知”。此区分不能用于自动猜测的假 track。

## 新 helper

`explicit_source_layers.infer_explicit_moving_source_layer(source, moving_support, ocular_donors, ocular_domain, noise, source_skin_prior=..., radius=6)`。

- `moving_support` 表示明确同动的细睫毛及其与眼睑相接的混合边，不是透明度。
- `ocular_donors` 是明确可见眼球源样本，不能与 moving support 冲突。
- `source_skin_prior` 复用已有源估计中的皮肤外观 proposal，不是真实分离皮肤层。
- 仍用冻结的 8 次 Huber 局部仿射面估计 B；foreground 候选颜色允许源 skin/lash 混合，纯源 RGB 三色 simplex 拟合 E。
- 在源层构建时将 E 限定于有界残差允许的可行区间；计算 `R=C−E*B`，不裁切最终输出、不运输量化误差。
- `estimated` 要求合法源供体、有限有界层以及可重现源 RGB；它不宣称真 alpha 已识别。
- `modelUncertain` 保留超出经验颜色模型尺度的估计；最终图像仍必须通过原独立质量测试及视觉检查。
- `unknown` 仅用于缺少可用底色支撑、域外或源层数值不合法等情况，依赖它的编辑必须拒绝。

在上述固定 27 像素标注上：`estimated=27`、`unknown=0`、`modelUncertain=1`、源 RGB 重建最大误差 0。独立最终 RGB 验收由材料合同测试执行；这些源合同数值本身不代表编辑效果已经可用。

新 helper SHA256：`0ccc00567ab09ce9fe2d2052fe11cad3635c30bb4926e16bdd853c1de098ec73`。旧 `robust_source_surface.py`、V5/V6/V7 均保留冻结。

只读证据：`test/results/eye-closing-source-20261119/explicit-moving-helper-diagnosis.json`、`explicit-moving-body-label-diagnosis.json`。所有图像/数据保留在 ignored 测试结果目录。
