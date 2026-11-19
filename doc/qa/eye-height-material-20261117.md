# EyeHeight 材料准入审计（2026-11-17）

## 结论

当前缓存图的 `feature.adjust` 请求可以执行 `eyeHeight ±15`，但参数图没有 `eyeGeometry`、虹膜边界或上下睑曲线。现行实现因此把通用纵向 SVF 作用到整个 eye polygon，无法证明上睑、睫毛和虹膜属于同一可移动层。该入口暂不具备可用的源材料证据，不能放行到“可用”评分。

## 复现范围

使用 `test/results/face-core-fresh/*-source.png` 与对应 graph，改为 `feature.adjust` 并移除只属于 `face.adjust` 的 `subjectMask`，分别渲染 `eyeHeight=-15/+15`。结果写入本机 `/tmp/eyeheight-replay-20261117b/`，没有进入仓库。

| 样片 | 档位 | 改变像素 | eye polygon 内 | eye polygon 边界 2px 内 |
| --- | ---: | ---: | ---: | ---: |
| portrait-makeup | -15 | 2716 | 869 | 220 |
| portrait-makeup | +15 | 2755 | 851 | 221 |
| portrait-hair | -15 | 896 | 260 | 127 |
| portrait-hair | +15 | 924 | 252 | 126 |
| portrait-roll | -15 | 1642 | 539 | 176 |
| portrait-roll | +15 | 1613 | 484 | 173 |
| portrait-woman | -15 | 739 | 129 | 92 |
| portrait-woman | +15 | 745 | 106 | 75 |

边界内变化只能证明有运动，不能证明上睑/睫毛的材料身份或自然度。当前缓存图的 `semanticMasks` 只有 `leftEye/rightEye` 多边形，没有可用于安全重塑的 ocular source layer。

## 候选

`core/image-core/src/eye_material_admission.rs` 提供隔离的保守准入：仅允许位于 aperture 内、远离边界并属于足够大连通分量的候选源材质；贴边睫毛/眼线和孤立高光被拒绝。3 个合成反例通过：

```text
cargo test --manifest-path core/image-core/Cargo.toml --release --lib geometry::eye_material_admission
```

后续只有在检测服务提供 canthus、上下睑有序曲线及巩膜/虹膜证据后，才能将 `eyeHeight` 接入候选材料层；证据缺失时应明确拒绝或提示，而不是继续放大通用形变。
