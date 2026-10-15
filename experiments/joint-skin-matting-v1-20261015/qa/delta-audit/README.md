# 差分合成独立审计

`test_delta.py` 保留已知真值的合成反例及原图精确 replay。9 tests 中 4 pass、5 fail，退出状态 1；未隐藏失败。

数学上 `C'=C_source+delta` 等于 `C_target_model+R_source`，`R_source=C_source-C_source_model`。这在真分层已知时与三层 oracle 一致，在零位移时即便层完全错误也恒等；零位移通过不代表分层真值正确。原图±100 PNG 精确重放均0差异，残差恒等误差<1e-8 byte。

错误 alpha/RGB 下的失败是质量鲁棒性边界，**不是宣称 compositor 能无信息恢复真值**。尤其 “inverse_colors_alpha_error_no_clipping” 名称不准确：原样本包含低 alpha 的 clipping/fallback。原结果保留不改，正确限定 alpha 范围后的 no-clipping 控制在 `experiments/alpha-feasibility-skin-v1-20261015/qa/report.json`，该控制确实达到常量已知背景下输出精确，虽然 alpha 仍不是正确值。

原图固定残差最大97.21byte位于耳根 `(652,430)`，但当地位移只有约.043px；不能只按残差最大值判断移动鬼影。+100另有下耳侧 `(630,503)`：direct alpha=.9740，经旧/新交界混合降为.27475，位移8.72px，writeWeight=.98197，固定残差88.02byte；`(631,501)` direct=.9706、mixed=.2934，位移8.92px，残差83.57byte。这些点提示 alpha 交界混合可能制造新的源分层不一致。

这些数值是模型固定残差，不是真值误差。已有完整图像和坐标记录于 `report.json`、`real/`；没有改候选或生产。
