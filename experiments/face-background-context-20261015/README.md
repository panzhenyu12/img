# 独立背景层的材质上下文消融

状态：五项真实 LaMa 对照完成；完整修图均未验收，未进入生产。固定 V36 原 source/graph/geometry，使用现有已校验 LaMa，单独改变模型可见的上下文及局部尺度。不宣称这是新模型或 SOTA。

此前直接修补合成人像窄条带，会把原来的脸颈当作上下文，生成肤色并抵消瘦脸。本实验将脸、颈、耳饰、衣物及其他前景排除，先生成辅助背景层，仅在真正露出且非冻结的像素下层合成，前景采样和透明度不变。

- `context.py`：源语义采样、材质上下文、精确 f32 合成。
- `run.py`：完整 crop 内 hair-only / hair-scene 两项消融、实际模型排除 RGB 扰动验证。
- `local.py`：固定 hair-scene known/write mask，局部 128/256/512 原分辨率推理。
- `full-context-review.md`：A/B 失败与源 person mask 截断问题。
- `local-context-review.md`：局部尺度结果与限制。

重放（需要新的输出目录，不允许覆盖证据）：

```sh
services/background-worker/.venv/bin/python experiments/face-background-context-20261015/run.py \
  --diagnostic experiments/face-material-fill-v1-20261015/woman-100 \
  --output /tmp/face-context-new
services/background-worker/.venv/bin/python experiments/face-background-context-20261015/local.py \
  --diagnostic experiments/face-material-fill-v1-20261015/woman-100 \
  --ablation /tmp/face-context-new --output /tmp/face-context-local-new
```

运行前精确 f32 采样缓冲区逐字节复现 V36，差异 0。所有候选均验证圈外、冻结和 alpha 变化 0。局部块能续出部分发丝，但旧亮细线与新皮肤阶梯仍在，整体未达到 8 分。仅一张固定原照，不能视为多图、不同幅度或浏览器验收。
