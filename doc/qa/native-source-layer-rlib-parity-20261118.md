# 普通 rlib 源层入口成片核对（2026-11-18）

这次核对检查 `image_core::adjust_face_with_source_layers` 是否能从正式 rlib 入口重现已经冻结的 source-domain-continuous 成片。没有改动阈值，没有重跑检测或模型，没有把测试模块源码复制到 example。

## 输入和复核文件

- Example：[face-source-replay.rs](/Users/modao/work/img/core/image-core/examples/face-source-replay.rs)
- Spec：`/tmp/face-replay-20261117/ordinary-rlib-48-photo-replay-spec.json`
- 输出：`/tmp/face-replay-20261117/ordinary-rlib-48-photo-replay/`
- 汇总：`rgba-parity.json`
- 执行指纹：`execution-fingerprints.json`
- 参考 PNG 使用已有 `neural-source-pair-more-alpha255`、`lupita-complete-source-context`、`neural-pair-upper-contours`、`neural-pair-other-contours` 冻结目录，未重新生成。

## 结果

| 项目 | 数量 |
|---|---:|
| 计划成片 | 48 |
| rlib 成功 | 48 |
| RGBA 完全一致 | 33 |
| 只有单通道 1 级差异 | 15 |
| 超过 1 级差异 | 0 |
| 失败 | 0 |

上脸三项（`forehead`、`foreheadWidth`、`temples`）在 user/dark 的正负档位共 12 张全部完全一致。它们经过 `promote_with_source` 的递归路径，仍使用同一 `FaceSourceLayers`，没有退回旧局部脸片。

15 张低幅差异均是单通道 1 级，未出现几何位置或结构性像素差。Lupita ±60 及 woman +60 的独立 alpha 对照证明，旧入口的归一化中心双线性 alpha 采样足以产生这些量化差；将旧采样作为唯一变量后，输出恢复到 0 差异。因此普通入口保持直接 full-canvas f32 alpha，不为了复刻测试历史误差而改算法。

## 边界

这是冻结输入的 API 迁移回放，不是任意照片的质量验收。它没有覆盖检测、模型、浏览器/WASM、身体和五官全功能、多控叠加、侧脸遮挡或实时交互。每张普通 rlib 成片的 `milliseconds` 是当前机器下算法调用计时，受尺寸、路径和系统负载影响；中位数约 9.56 秒，不能宣称实时性能或产品已可用。
