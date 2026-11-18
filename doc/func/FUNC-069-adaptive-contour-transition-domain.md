# FUNC-069：自适应真实过渡域候选

状态：隔离研究候选；未接入默认人脸形变路径。

## 目的

当前轮廓投影只看 active contour 的截断区间。固定连接点或检测噪声落在区间边界时，平滑项会把不可连续的运动集中到边界，表现为轮廓折角、裂缝或局部不响应。该候选把原始轮廓上真实存在的零容量邻域一并纳入求解，让运动在观察到的皮肤上下文内连续衰减，同时保留硬附件/固定材质的零运动约束。

## 算法契约

`core/image-core/src/face_displacement_curve.rs` 提供：

- `extend_zero_capacity_context`：沿源闭合轮廓扩展 active arc，直到两端各有两个零容量样本；不插值、不新增容量，无法找到零容量端点时要求调用方改用周期求解；
- `project_context_curve`：在扩展后的源顺序曲线上做位移 fairing，保留 core 的物理 bending length，避免扩大观察域后自动放大编辑；
- 已有合成回归覆盖双向符号、缩放/平移交换、固定点、非连续 core 和无零容量端点等反例。

附件与用户保护区域必须作为 exact fixed samples 或 source material mask 传入；仅有轮廓 JSON 时不能推断隐藏皮肤、头发或背景归属。

## 离线验证

`core/image-core/examples/face-context-transition-replay.rs` 对 user、makeup、dark、woman 的正负极值曲线做几何回放，结果见 [metrics-real-context.json](../../experiments/face-context-transition-20261117/metrics-real-context.json)。旧 area replay 的角度公式错误（遗漏叉积且 atan2 优先级错误），因此撤回 dark “397–842° 降至 11–14°”结论。新 context 的角度实现已正确，但旧比较不可用。面积响应误差为 0.065%–0.494%，且部分相邻位移跳变退化；没有实图无裂缝验收。

## 晋升条件

该候选只有在 source material mask 能证明可动皮肤域、硬附件固定域和连续连接关系，并且全部 holdout 同时满足面积响应、局部位移跳变、额外转角和图片 crop 无裂缝复核后，才可考虑接入 `face.rs`。当前实验只证明过渡域比截断 active arc 更有研究价值，不能证明图片级自然度，因此保持隔离。

验证命令：

```text
RUSTFLAGS='-Awarnings' cargo test --manifest-path core/image-core/Cargo.toml displacement_curve::tests -- --nocapture
```
