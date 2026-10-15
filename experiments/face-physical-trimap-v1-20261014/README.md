# 脸部物理 trimap 候选 V1（拒绝）

2026-10-14。以生产源码快照、用户 `user-face-contour-20261014` 原图和固定 `graph100.json` 做隔离实验；只增加连接脸/颈边缘的物理 RGB color-line trimap 恢复，没有修改生产 `face.rs`、`face_matting.rs` 或公共 WASM。

候选恢复了与已观测脸/颈组件相连的语义零边缘，且要求 selected-person confidence、RGB 前景/背景色线、局部 skin/neck 邻接同时成立。头发、耳朵、饰物、另一人物、用户保护和 frozen 像素均禁止恢复；透明 RGBA 直接保留原 alpha。`face_matting::tests` 的 7 项测试全部通过。

## 失败证据

固定 renderer 回放见 [`report.json`](report.json)，原图/生产基线/候选对比见 [`evidence/face-comparison-1x.png`](evidence/face-comparison-1x.png) 和 [`evidence/island-comparison-10x.png`](evidence/island-comparison-10x.png)。目标旧皮肤岛为 `x=407..408, y=547..552`；候选将其中多个棕色像素 `[85..128,70..104,63..94]` 变为约 `[25..27,15..18,11..13]` 的背景色。候选输出与生产基线有 439 个像素变化，物理 alpha 新增 54 像素（包围盒 `x=381..659,y=497..640`），远超目标 12 像素岛。

根因是恢复出的 `color_alpha` 进入了可移动层和 hole/reconstruct 路径。逆采样在这些恢复像素处找到零前景，背景估计器随后填入暗色背景；这不是有真实皮肤 donor 的可靠重建。虽然 frozen/known-scene/ROI 外/输出 alpha 均保持 0 变化，视觉上的暗洞和恢复范围仍不能接受。

**结论：候选拒绝，不进入生产。** 后续若继续实验，必须把恢复 alpha 与几何可移动支持分离，并证明每个恢复像素有真实前景 donor；单纯扩大 alpha 或把暗色背景当作修复结果不满足验收。

## 重放

```sh
CARGO_TARGET_DIR=/Users/modao/work/img/.run/face-physical-trimap-v1-target \
  cargo test --manifest-path experiments/face-physical-trimap-v1-20261014/Cargo.toml \
  --release face_matting::tests
/Users/modao/work/img/services/background-worker/.venv/bin/python \
  experiments/face-physical-trimap-v1-20261014/replay.py
```
