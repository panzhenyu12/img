# RBF/SVF v17：selection、垂直参数和情境保护回归修复（未验收）

此阶段只在隔离 crate 中运行。生产人脸与 public WASM 未改。

selection / protect / ROI 在请求目标、可行目标、速度采样、QP 固定标记和 shooting 分母中采用同一 external constraint。`chin` / `forehead` 有垂直请求时跳过锁定 v 的侧曲线 QP，避免把垂直功能取消。零 selection、全 protection 图像逐字节一致；半脸 selection 只改变选中半脸；chin=100 确实延长至少 9px。

完整 face 套件首次出现一个真实回归：邻近独立人物被作为耳根 attachment，连带使旁边脸部不动，x43 仍为源脸 RGB。没有删除或放宽原断言。最终把全固定域与实际耳/发/饰品连接域分开：只有连接域限制可行脸缘；其他固定情境只在自身域内连续归零，允许脸缘缩离它。自脸边缘误冻结纠正只用于独立 head/face 分类支持，避免把普通手工或同一面具重复存储当作解析证据。新暴露 x43..45 为干净场景，绿色独立人物每个源像素保持 exact。

中间版本给情境与连接重复加 gate，改变了原图耳侧过渡。该版本完整保留在 `rbf-v17-first-context` 及 `face-first-context.rs`；最终 gate 只作用于非 attachment 固定情境。

最终完整 face：**39 passed / 0 failed / 1 ignored**（日志 `unit-test.log`）。同一用户原图 ±30/±60/±100 六档与 v16 PNG SHA **全部逐字节相等**，因此它是回归修复，不能被声称为颈部视觉改善。固定 ears/accessories/hair 0 改变、9 个实际五官内部 0、alpha 0、纯收缩 scene 背景 0。`report.json` 只把前128个轮廓控制计为 boundary，排除五官控制。

总体仍拒绝：六档 face/neck 黑楔、材质硬接及旧阴影残留继承 v16，未达到 8/10。无 real-photo 九参数全面验收、无 WASM parity 与交互性能验收。数值无 fold 不构成视觉评分。

完整可重建源码与依赖：`experiments/face-rbf-v17-20261014/src`、Cargo.toml/lock；从生产基线的完整 patch、上下文修复 patch、全部输入/二进制/六档输出/诊断 SHA 见 `provenance.json`。初始失败记录见 `regression-history.json`。材料 probe 的证据边界见 [颈层负例](../face-amodal-neck-20261014/review.md)。

独立 target 从最终归档源码重建 native renderer 后，六档 PNG 与原冻结候选再次全部 byte exact；见 `independent-rebuild-output-parity.json`。这验证源码/测试/渲染证据对应，仍不改变视觉拒绝结论。
