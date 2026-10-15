# 用户耳饰边缘：model-free scene donor 复核（2026-10-15）

**结论：拒绝接生产，尚不达可用标准。** 两种同侧背景取样方法都没有消除 +100 的旧肤色条或 -100 的黑楔/锯齿。只修改背景颜色无法修复原 source skin support 漏掉的可见轮廓。

实际输出保留在：

- `experiments/accessory-skin-decomposition-v1-20261015/scene-donor-v1/`：3 尺度 coherent patch donor。
- `experiments/accessory-skin-decomposition-v1-20261015/scene-donor-v2-local/`：最近同侧已观测背景边界，9 列中值、纵向多尺度、3 列留出验证。
- 两目录均含 ±100 的全脸 1×、两侧耳饰 1×、target/donor/protection overlay、差分、完整候选与报告。
- 详细说明：`experiments/accessory-skin-decomposition-v1-20261015/scene-donor-README.md`。

冻结 V36 inverse、v2 source layers、corrected alpha 与金属前景。只在 v2 原 write mask 内、独立 source skin 支持且实际露出 backing 的区域写入；耳朵、头发、soft/opaque metal、unknown、insufficient 均不允许改动。旧 semantic accessory 区包含误标皮肤，若整区冻结则两方向 source strip 可写域均为 0；本实验保留该 no-op 控制组，实际以独立证据处理误标。

| 指标 | +100 | -100 |
|---|---:|---:|
| 可写旧条像素 | 216 | 39 |
| v1 实际变化 | 213 | 29 |
| v2 local 实际变化 | 213 | 30 |
| 保护区/目标外变化 | 0 | 0 |
| target 边界均值梯度：原 v2 | 23.55 | 10.51 |
| target 边界均值梯度：v1 | 73.61 | 10.92 |
| target 边界均值梯度：v2 local | 82.06 | 11.61 |

v1 的 4 个 component/direction 均被颜色门禁拒绝。v2 local 用到 1,836 个完全排除人物的 donor；右侧已观测留出边界 RMSE 3.75，左侧 10.26。但隐藏背景离 donor 中位数 44px，已观测拟合不代表未知区真实重建。1×检查也明确失败，不能包装成 8 分。

```sh
services/background-worker/.venv/bin/python experiments/accessory-skin-decomposition-v1-20261015/test_scene_donor.py
```

实际 4 行输出检查通过（区域/alpha/inverse/source/donor 保护，以及故意污染样本必须被拒绝）；所有候选 `accepted=false`。没有改 Rust/WASM/Go/前端，没有重启或切换生产服务。
