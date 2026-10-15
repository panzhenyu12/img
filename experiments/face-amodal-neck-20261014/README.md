# 遮挡颈部延拓：隔离研究，未接受

日期：2026-10-14。受 ICCV2025 Inpaint4Drag 的 warp/inpaint 分工启发，针对实际用户照片新增本项目的层归属实验。没有复现或声称发明新的神经网络；可见颈侧拟合、最近观察纹理和速度流等组件本身不是新方法。本实验用于验证“连续形变以外还必须预测正确下层”这一任务假设。

基线是 v15 的同一源图、检测、目标与不可变 native binary。`instrumentation.patch` 只新增层标签、holes、crop、重建前后输出；其 PNG SHA 与未插桩 v15 完全相同。

1. 从原图可靠颈部观察提取侧边，变换到双眼定义的坐标系，稳健拟合局部侧边，仅在有限遮挡范围延拓。
2. 用延拓颈部 envelope 约束本应露底的候选，只有 holes 且不冻结的位置可写入。用观察到的颈部纹理作基线，按已经传输的 face alpha 重合成；没有对可见脖子/全图做拉伸。
3. 单独撤销旧颈部 relighting，保存消融。这个输出不混称只改层归属。
4. 将上述 underlayer 送入现有 LaMa，仅对 neck holes 重建并重新合成。LaMa **没有输入颈部语义标签或解剖先验**；颈层输入/空间写权限不构成神经语义保证。

结果：颈侧假设改变2,596像素，大黑楔缩小，但细缺口、硬阴影和纹理复制仍失败。LaMa目标3,339像素，结果改变2,912像素，在允许的颈层内仍生成深色条带，失败。两组圈外/冻结/原RGBA alpha变化均0，这只能证明写权限，不是画质通过。不接入生产，不评分到8。

本轮是同一1024²原照的派生实验，没有独立 amodal alpha 或 hidden-neck 纹理真值，不代表其他照片。以后必须加可见颈侧的物理边界、face/neck联合轮廓、真实固定连接、材质条件模型和有clean参考的遮挡实验。

完整记录及图像：[QA](../../doc/qa/face-amodal-neck-20261014/review.md)。源码版本、native输入/输出和官方模型哈希保留在该报告，不把本机CPU时间当论文GPU基准。

## 复现

需要 v15 隔离 crate、记录的用户原照/检测，以及 SciPy/Pillow/NumPy。将插桩 patch 应用到隔离副本，构建 `image-render`，设置 `FACE_FLOW_DEBUG_DIR`，回放同一 `faceWidth=100` graph。原照+graph不变，插桩结果须先与冻结v15逐字节对账。

默认本地插桩证据目录为 `test/results/face-material-v1-20261014`，不会读取编辑后图像重新检测。

```sh
services/background-worker/.venv/bin/python experiments/face-amodal-neck-20261014/run.py --output test/results/face-amodal-neck-replay
services/background-worker/.venv/bin/python experiments/face-amodal-neck-20261014/refine.py --evidence test/results/face-amodal-neck-replay
```

必须使用新目录，保留冻结输出。模型权重为现有 LaMa TorchScript，第二条可选；没有模型时第一项几何/材质诊断可独立运行。这些不是产品操作接口。
