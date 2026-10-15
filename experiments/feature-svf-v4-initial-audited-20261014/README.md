# V4 初次接受版本：已由 V4b 修复替代

这里保存 2026-10-14 初次接受的完整 `geometry.rs`、生产差异和独立审计失败证据。源码 SHA256 为 `c68070022e8b37e7d9ea250239a7caf305d50f6e536836cd00d1ab480821f9ad`。这不是当前生产版本，也不是完整可独立构建的仓库。

后续独立审计发现 `eyeTilt±100` 的眼中心 forward 局部面积额外放大约 9.34%，以及 identity 采样丢失 alpha=0 像素的 hidden RGB。此前正 Jacobian、固定坐标和恒定 alpha 测试没有验证这两个语义条件。当前 V4b 用正确的旋转生成元与 identity 原字节复制修复，增加独立面积/角度和 public graph 可变 alpha 回归。

旧接受报告保存在 `doc/qa/brow-flow-v4-20261014/initial-accepted`；移目录后18个相对链接已修正，原字节另存`review-original-relative-links.md`并记录link-relocation hash。旧报告的全仓检查和artifact SHA只证明旧版本。新版本复核见`doc/qa/brow-flow-v4-20261014/review.md`和`hotfix-v4b`。
