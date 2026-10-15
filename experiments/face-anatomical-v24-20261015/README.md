# face anatomical v24

以 v23 为基线，移除颈部密集零控制点，改为 join 后的侧向速度局部 quintic taper；外颈 guides 保留有限独立预算，避免 RBF 大半径造成黑色凹口。仅实验，不改 production。

## 结果（2026-10-15）

- 变量：移除 join 下方密集 zero controls；脸部边界横向速度在 join 后 0.24×faceHeight quintic 衰减；joint_neck 调用 `restore_neck_lighting`。
- 回放：ZIM frozen source support，`faceWidth ∈ {-100,-30,0,+30,+100}`，原尺寸 1024×1024。
- 几何指标（+100）：`folds=0`，`minJac=0.3938`，`maxForwardInverseErrorPixels` 由 flow debug JSON 记录。
- 视觉判定：失败。+100/-100 颈部仍有下颌下方棕色横向色块和两侧硬折；没有进入 production。
- 测试：实验 crate 可编译并运行 image-render；完整 `cargo test --lib` 曾启动但因 debug 回放耗时被外部终止，不能宣称全量通过。
