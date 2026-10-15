# v19 一次联合 face+upper-neck 假设（隔离，未验收）

可重建完整crate `Cargo.toml` / `Cargo.lock` / `src/`。v17源face目标/RBF/SVF不变；现有raster skin/neck+face label提供有限启发联合seed，RGB color-line refine alpha，Germer只估F/B颜色。显式固定障碍不穿，high-alpha未连通skin岛不纳入；可见upper-neck同流运输，下颈运动连续falloff，原ROI/mask不扩大。不增加necklace/class原始概率协议。

隔离native开关 `FACE_JOINT_NECK_EXPERIMENT=1`；公共协议/production/WASM不扩展。`replay.py`从冻结candidate binary对同一canonical/source graph运行±100/±30；环境manifest、来源SHA见QA。最初CLI额外params被validation拒绝，之后改用native env，保留历史记录。

功能测试快照：`face-tested-before-roi-debug.rs`，env更新后完整face40pass/0fail/1ignored；该测试通过内部params直接调用分支，env CLI路径由四档图像覆盖。最终 `src/face.rs` 在该快照后仅增加ROI真实mapper trace，`replay_roi.py`验证四档PNG byte exact。没有对trace源码虚报测试重跑。复现可在独立target运行：`cargo test --release --manifest-path experiments/face-joint-v19-20261014/Cargo.toml --target-dir .run/face-joint-v19-reproduce --lib face::`；native render需前述env。

结论：旧大黑楔明显改善，source-face实际约20px，联合outer alpha .5边缘也移动；仍12px源皮肤岛、耳根点/硬边，未8、未生产。同源图有AI水印，不称验证相机实拍或多原片。冻结一次候选，不反复阈值调试。

[主QA/来源/四档输出对账](../../doc/qa/face-joint-v19/review.md)，[root独立目视](../../doc/qa/face-joint-v19-root-review/review.md)，[只读因果分析](../../doc/research/face-upper-neck-joint-alpha-causal-20261014.md)。
