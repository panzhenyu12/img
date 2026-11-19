# 新源闭眼实验回放

`run_closure.py` 接受原图与一次新的完整 MediaPipe 观测，从该输入重新生成眼睑曲线、眼/眉 polygon 和 face region，再调用指定的现有 native 场生成器。不读取历史样片的 state、graph 或 field，不重新编译整个工程。此入口没有开放公共 `feature.adjust`，也没有覆盖睁眼、戴眼镜或任意照片。

```sh
services/background-worker/.venv/bin/python experiments/eye-closing-source-20261119/run_closure.py \
  --source /absolute/path/original.jpg \
  --observations /absolute/path/observations.json \
  --output /absolute/path/new-output-directory \
  --field-runner /absolute/path/eye-field-tests \
  --estimator module_name:estimate_observed_ocular
```

`--estimator` 必须显式指定，避免将候选当成默认稳定算法。输出目录必须不存在。多脸观测必须用 `--face-index` 明确选中一个人；EXIF 旋转源必须先显式归一化并重新观测。本实验只支持不透明 RGB；非全不透明源拒绝，不无声丢弃 alpha。

固定流程：

1. 校验尺寸、完整观测与相邻 `report.json` 的原图/观测 SHA256 绑定（存在时）。
2. 每眼只准备一次源曲线和材料估计，在生成任何目标 field 前完成。
3. 为 `0 / -15 / -30` 分别构造新 graph，同一份源准备复用三次。
4. 验证 native 零值 RGBA 恒等、非零 field 的 donor/翻折/反演诊断、材料支持和 RGB 有界性。失败保留原因，不自动减小强度或回退输出。
5. 保存全图、相同范围的眼部裁图、源层、native 单平面对照与输入/代码/field hash。运行前后代码、源图、观测或 native 可执行文件变化时拒绝本次归因。

native 可能使用不同的 JPEG 解码器。因此 native 零值恒等断言与 Python 合成零值恒等断言分别记录，解码差值也会报告。源分层与最终合成始终使用同一个 Python RGB 解码结果。

源曲线使用物理像素中心 `(x+.5,y+.5)`，冻结 native 场使用整数 pixel-index。新 graph 的特征点显式转换一次：`(MediaPipe normalized × 原图尺寸 − .5) / 原图尺寸`。原始观测、源曲线和输出 inverse 不修改，转换后越出像素中心范围的点拒绝，不 clamp。外围 region 保持原始 normalized FACE_OVAL bounds 加 20% padding 的准入边界合同，不套用特征点转换。此适配不修改旧样片的历史 field。

未提供 hair/ears/accessories 固定遮罩时，冻结 native 代码使用空 stationary 向量，lookup 默认 false。runner 明确将 0 字节 stationary 转为全 false；任意非零但不足整幅的字节数仍拒绝。

`report.json` 的 `rendered` 仅表示生成和数值检查完成，不表示高光保留、睫毛只移动一次、轮廓自然度或新样本可用性通过；这些必须另做独立全图/裁图审查。源分解满足有界约束也不能证明单 RGB 图像中的光学 alpha 唯一可辨。

输入与场合同回归：

```sh
services/background-worker/.venv/bin/python -m unittest discover \
  -s experiments/eye-closing-source-20261119 -p test_run_closure.py -v
```

`test_observed_ocular.py` 包含独立构造的虹膜亮暗纹与跨虹膜睫毛反例，用已知分层真值检查材料估计。不能为候选通过而改反例或放宽阈值。

`--materials /absolute/path/materials.json` 可显式提供绑定原图的人工材料身份修正。源准备仍在所有 field 前完成一次，并以同一场另存自动控制组和人工候选；unknown 依赖会拒绝。格式、人工与自动的能力边界及反例见 `MANUAL-MATERIALS.md`，不能将人工结果报告为自动算法通过。

2026-11-19 的冻结 V5 已完成三张新源照片的有限视觉审查，记录在 `doc/qa/eye-closure-fresh-source-20261119.md`。指定输入的 −15/−30 外观成功与 V5 仍失败的已知真睫毛反例同时保留；不代表公共功能开放或整体达到 8 分。
