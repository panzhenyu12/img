# v17 未验收隔离源码

完整 crate 已保存，可用 Cargo.toml 重建。`src/face.rs` 为最终上下文保护修复；`face-before-context.rs` 和 `face-first-context.rs` 保留失败中间阶段。它们不属于生产模块。

复现：`cargo test --release --manifest-path experiments/face-rbf-v17-20261014/Cargo.toml --lib face::`。`replay.py` 使用冻结 `.run/face-rbf-v17-candidate-image-render` 和 canonical 原图/graph，不重新检测编辑后的图。

审查、回归结果与 SHA：[v17 QA](../../doc/qa/face-rbf-v17/review.md)。
