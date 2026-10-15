# ZIM 整个人像覆盖实验

状态：用于隔离整体脸型的源覆盖问题，未接入编辑器。复用项目已冻结的 ZIM ICCV 2025 官方 ONNX 与适配器，没有新下载模型。

此前分别估计皮肤、耳朵、头发会引入内部接缝。本次直接以原始照片和原始解析生成皮肤、头发、耳朵、饰品、颈部、衣物的正点，背景负点必须在全部人物类别外；目标是整个可见人物。只读原图，不在修改结果上重新检测。实例归属仍是独立问题，这个单人例子不能证明多人分割有效。

用户图的 point-only、point+box 各保留四张原始 float alpha，模型分数最高均为 index 2。原尺寸目视两耳、发际和颈部，比原类别解析少了部分背景暗边；这是 matte 可用性的探索，不是最终修图验收。`cutout` 只是原 RGB 乘 alpha 的可视化，并未完成前景颜色去污染。饰品的高置信核心覆盖不完整，也不能把解析标签当作真实 alpha 标注。

- [提示、模型/源文件哈希与全部候选指标](outputs/report.json)
- [左耳原尺寸](outputs/points/left-1x.png)
- [右耳原尺寸](outputs/points/right-1x.png)
- [固定几何的 Rust alpha 替换验证](../whole-person-alpha-override-v1-20261015/README.md)

原始推理约一秒多的 decoder 时间只对应本机 CPU 和这次输入；不是浏览器热响应承诺。运行 `services/background-worker/.venv/bin/python experiments/whole-person-matte-v1-20261015/run.py` 会拒绝覆盖已有输出。
