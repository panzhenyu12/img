# 用户耳饰 scene-only v5 独立验收

v5 的 128/256 两种 scene backing、`+100/-100` 共 4 行均通过 alpha、inverse geometry、写域外和冻结层重建；透明金属核心 2,332 像素与原图一致，独立源高光探针 1,027 像素全部保留。

它仍未通过最终效果：raw accessories 语义区域内有 +100 的435、−100 的1,112个实际变化像素，说明软边/附着/孔隙保护不足；原尺寸+100有新肤色/暗边，−100虽减轻黑楔仍有台阶。scene prior/LaMa 只是隐藏背景假设，不能视为真实遮挡恢复，也不能在两个方向里择优上线。

详细证据：[v5 review](../../experiments/accessory-skin-decomposition-v1-20261015/independent-audit/review-v5.md)；[独立报告](../../experiments/accessory-skin-decomposition-v1-20261015/independent-audit/report-v5.json)。
