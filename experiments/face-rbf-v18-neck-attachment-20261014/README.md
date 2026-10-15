# v18 单次颈部连接候选（拒绝）

完整crate源码与Cargo锁定文件已归档。可见neck短连续边缘在眼坐标系拟合，再与source faceContour求交；同一可行目标、固定gate和RBF/SVF逆映射。不包含用户照片坐标硬编码。不进入生产或public WASM。

效果：大黑楔减少，但颈影拼接、连接小黑点仍在，且+100实际最大位移仅8.15px（请求20.02px），拒绝。六档输出与数学/保护诊断完整保留。

`replay.py`使用冻结 `.run/face-rbf-v18-candidate-image-render`、用户canonical原照/face-graph，不重新检测编辑图。源码测试务必使用独立target，不能复用v17旧单测：`cargo test --release --manifest-path experiments/face-rbf-v18-neck-attachment-20261014/Cargo.toml --target-dir .run/face-rbf-v18-reproduce-target --lib face::`。

审查、原图因果对比、输出SHA见 [v18 QA](../../doc/qa/face-rbf-v18-neck-attachment/review.md)。
