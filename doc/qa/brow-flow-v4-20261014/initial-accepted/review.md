# 五官 SVF v4：眉毛眼睑保护与完整映射复核

接受本轮几何修复：原来眉峰/眉尾会把上眼睑拱弯；现在眉形仍改变，眼睑与眼线保持稳定。固定遮挡及硬选区边界的折叠、眉值跨零时切换眼睛算法也已修复。**不提高到 8 分**：强眉峰仍夸张、眉尾在靠近眼睛/遮挡时作用受限，周围皮肤纹理仍可能拉伸，组合导出也需数秒。

生产源码已冻结。独立 function 对账见 [FUNC-020](../../../func/FUNC-020-feature-svf-quality.md)，此前的错误效果仍保留在[原始功能审计](../../user-feature-audit-20261014/review.md)。

## 结果及证据的范围

| 检查 | 结果 | 能证明的内容 |
|---|---|---|
| 默认实际 f32 mapper 回归 | 30 个当前开放几何参数 ±100，另含组合/硬软边界/roll30/tall，共 82 场景通过 | 最终采样映射的数值行为、固定供体与位置、眼域、目标位移、实际像素和 alpha |
| 默认跨零回归 | 4 组，全图最终 mapper 变化 <0.01px，RGBA ≤1/255 | 同时调眼/眉/鼻时，0→0.001 不切换无关的算法 |
| 原尺寸照片实际 mapper | 42 例全部零折叠、固定 donor 导入 0、固定运动 0、眼包络偏差 0 | 此批实际 detector + 遮挡/选区/保护输入；不是 f64 内核或理论证明 |
| 实际编辑器 PNG 导出 | 19 例 native/WASM 全 RGBA 逐字节相同，页面错误 0 | 控件、工程恢复和实际 PNG 导出正确使用共享核心 |
| 原尺寸照片跨零导出 | 3 对 0→0.001，RGBA 变化均为 0、alpha 变化 0 | 1024px 用户原照上的稳定性；微小请求可被输出量化掩盖，不要求所有微小值都产生可见变化 |
| 最终 native 原尺寸对照 | 28 项眉峰/眉尾 ±100 +2 项 +40，alpha 改变 0 | 目标确实变化及眼睑修复可目视；自然度需单独评估 |

42 个照片 mapper 的最小采样 Jacobian 为 **0.0936301**，最大逆流误差 **0.00921631px**。Jacobian 使用 native 整数像素坐标与 h=0.25px 中心差分；检查整个活跃支持域包围矩形，包含组合的鼻矫正/对称支持域。零折叠仅是这些样本与采样位置的结果，不是连续空间或所有合法输入的严格证明。

照片来源合计六张独立原照：本次用户照片和五张既有缓存；`portrait-roll` 是带妆照片的旋转派生。多种参数、派生图和合成纹理不增加独立原照数量。[完整 native/mapper 报告](report.json)、[19 例 UI 报告](ui-parity.json)、[实拍跨零对比](../continuity-photo.json)。

## 原尺寸目视

先看三列 **Original / Baseline / SVF v4**。这些裁切保留源图像素；联系表只组织图片。用户照片 ±100 和 +40 已检查，另外六个照片版本的四种极值裁切逐张复核。

- [用户眉峰 +100](../user-browPeak-100-comparison.png)：原版上眼睑明显变成尖拱；新版本眼睑稳定，眉毛仍拱得很强。
- [用户眉峰 -100](../user-browPeak--100-comparison.png)：眉峰下降可见，靠近眼睑的运动受到约束。
- [用户眉尾 +100](../user-browTail-100-comparison.png) / [眉尾 -100](../user-browTail--100-comparison.png)：眉尾改变，旧斜眼皮不再出现。
- [用户眉峰 +40](../user-browPeak-40-comparison.png) / [眉尾 +40](../user-browTail-40-comparison.png)：温和档效果可见，眼睑稳定；不把两项温和档当作新原照。
- [各图眉峰 +100](../browPeak-100-all-samples.png) / [-100](../browPeak--100-all-samples.png)、[各图眉尾 +100](../browTail-100-all-samples.png) / [-100](../browTail--100-all-samples.png)。浓眉、笑纹与遮挡区域仍有局部纹理拉伸、响应不对称或受限。

## 为什么拒绝前面的候选

V1 把直接位移乘眼域 damping，眼睛像素保护测试虽通过，独立场检查仍发现 peak/tail 的负 Jacobian。V2 只保护眼裂，眼睑折线仍跟着眉毛弯曲，作为 Python f64 原型保留。

V3 把位移改成 RK4 流，用户照片四种极值的内核/眼域可通过，但**最终映射**仍含 stationary donor identity 返回和硬边界截断。42 例中 10 例折叠，`portrait-hair browTail=-100` 最小 Jacobian -27.73、37 个折叠像素；`portrait-makeup browTail=-100` 为 -36.07、21 个。完整补丁、源码、诊断、日志与三项 native 失败裁切保留在 [V3 撤回目录](../../../../experiments/brow-flow-v3-rejected-20261014)。

最初 V4 只在眉毛非零时启用 SVF，因此 `eyeSize=100 + browPeak 0→0.001` 使眼睛由直接位移切成积分，默认测试失败 **5/255**。这个候选也保留在 [V4 跨零失败目录](../../../../experiments/brow-flow-v4-zero-crossing-rejected-20261014)。最终所有有语义的五官几何均使用 SVF；颜色/妆容与无语义手工模板继续各自原路径。

## 复现和编译

[小检测 fixture](../../../../core/image-core/tests/fixtures/user-brow-geometry-20261014.json) 是实际 user detector 坐标；原始 JSON 的 raster 不纳入 fixture，固定障碍由测试显式生成。普通测试生成纹理，无 `test/results`、图片下载、optional 模型或 env 依赖。测试 observer 仅在 cfg(test) 编译，通过 thread-local 控制，不写生产版本的 QA/env 钩子。

```sh
cargo test --release --manifest-path core/image-core/Cargo.toml geometry::tests -- --nocapture
make native wasm
```

本轮 geometry 模块最终 **27 passed / 0 failed / 1 ignored，16.45s**。ignored 为需本地照片路径的专项诊断，不是默认验收；42 项已单独运行并完整保留在报告里。native 和 WASM 均构建通过。全仓 Rust/clippy/Web 由主任务继续统一核对。

可选照片/页面复核由 [native 回放脚本](../../../../scripts/audit_brow_flow_quality.py) 和 [真实 UI 脚本](../../../../scripts/audit_brow_flow_ui.mjs) 完成，需要报告中声明的可选缓存和不可变 worker；它们不取代默认单测。

42 例 mapper 报告产生于把 SVF 开关扩大到全部有语义五官之前；所有 42 例都有非零眉参数，所走的速度/运动域/最终映射没有再改。最终 worker 重新渲染 30 项眉毛 PNG，SHA256 全部与初始 V4 相同；19 例实际 UI 检查额外覆盖新的单项眼/鼻/嘴和跨零路径。[源码/输入/fixture/worker/报告哈希](provenance.json)。

## 性能和未验收项

本轮最终 native 总进程时间408–1559ms，包含图像 I/O；此前同批约 477–2520ms。并发负载不同，不能据此声称算法变快。本轮 UI 导出 1839–10761ms，并发全仓测试和预览队列均影响时间。这不是稳态热帧测量，不能称实时。尤其多个局部场同时开启会增加 RK4 步数，后续可研究网格积分/插值、局部域和缓存，但不能删减当前完整映射或跨零检查。

目前眼睛保护包络、冻结蒙版与参数场仍是二维启发式，没有三维语义重建、真实发丝 matting 或新皮肤/眉毛材质合成。极端形变无折叠不等于自然。4K、严重遮挡、侧脸、复杂妆容、密集同部位组合、短距离硬保护及多分辨率一致性仍需扩大验证；其他五官虽经过数值极值检查，不能据此提高其自然度分数。

## 冻结与产物 SHA256

| 产物 | SHA256 |
|---|---|
| geometry.rs | `c68070022e8b37e7d9ea250239a7caf305d50f6e536836cd00d1ab480821f9ad` |
| 检测 fixture | `fbe960b7b04aee2c6b756b62aa0df0f5b1b7550163bbbd740cf10459b24b096b` |
| 用户 canonical PNG | `5f15e92b14c44a468d924af5ac985a7cc7dc1c87d8ab235f2c94b2e88b173c73` |
| 最终不可变 native | `b1de7be834980de76803f53de0e163e223ee6f6f41fdb2c2430ccdcb95ee5203` |
| WASM | `24dfd715f2c2612f4b8f4b9e1eff4a1c68860cfdb28e0f0933f30f260e2c14c6` |
| native/mapper 报告 | `f049498ea05dfc4281e5a6278983c44281c6602e79feb5b2fbc8bf2ebdef8b47` |
| UI 对账报告 | `d189c0e91cda197ac5813b630b467ba093ef779ac14b6c765af3edc1da68ec6c` |
