# FUNC-070：源材质统一运动域候选

状态（2026-11-17 续）：**隔离候选已获得限定场景的离线可用证据，尚非通用可用或逐功能 8 分，未接入默认路径。** 正确源资料下，user、woman、dark 与 Lupita 的脸宽 `±30/±60` 共 16 个结果经独立原尺寸审查，未发现所查裂口、旧轮廓、颈条带或耳钉断连。Lupita 必须保留同源附件运动计划。fresh hair 的约 1px 接点及其他姿态/功能限制仍保留；旧失败结果不删除。

## 目标

使同一可动材质使用连续的逆向采样场，并区分三件事：速度能在哪里求值、源图实际观察到什么材质、目的像素允许显示哪一层。原皮肤二值覆盖不能兼任速度场定义域；否则扩张走出旧轮廓即遇到零速度，相邻目的像素可能采到同一源位置。

速度延伸本身不修改物理 alpha、源颜色、背景供体权限或输出保护。硬衣物和显式保护仍是运动障碍；头发、耳朵和饰品的源层及附着关系由各自既有合同决定，不能仅因为延伸速度就自动改成可动皮肤。

## 实现

- `core/image-core/src/face_joint_domain.rs`：原 `build` 保留按源角色赋权的基线；亚像素查询改用双线性插值，无硬障碍的有效域保留完整权重，不再回退旧附件衰减。
- `build_with_motion_halo` 在已观察可动材质外定义速度延伸带，再以 quintic smoothstep 衰减；传播不穿过硬障碍，裁剪外缘速度归零。四邻域距离配合方向余量覆盖斜向运动。离散插值不等于全局 C¹ 连续或正 Jacobian 的数学保证。
- `experimentalFaceProfile: "source-domain-continuous-v1"` 作为测试候选调用该延伸域，当前行程预算为原轮廓最大请求位移范数的两倍。它是工程预算，尚未证明覆盖任意组合的实际积分路径；应以真实流场和成片检查为准。
- `face_profile_replay` 用相同源图和 graph 回放候选；连续域、静态控制消融和既有材质路径恢复分别保留输出，避免把多个改动的效果混为一项。
- `attachment_fixed` 既是输出锁，也是统一运动域的连续障碍；`velocity_weight` 跳过的是重复的旧附件距离 ramp，并非取消真实固定附件边界。上游已准入的 shared hair 不再属于该固定集合。

## 已确认的因果链与修复

1. **脸/颈语义边界被当作输出截断。** 速度场已连续，但 connected neck 的像素被硬写入门槛跳过，形成同一皮肤材质两侧采样不一致。先用解析 RGB 梯度及真实逆采样做回归复现，再删除该门槛，完成红→绿验证。保护区和真实遮挡继续由原合同处理。
2. **零附近更换了已经可见的配对底图。** 预乘前景残差仍与源底图配对，但合成时整块换成重建目的底图，即使微小位移也产生有限色跳。现仅把正覆盖差对应的新显露部分交给重建底图；既有 shared-cheek/periodic-curve 跨零反例的最大 RGB 变化由 4 级降至 0。这是局部最大重叠近似，不是透明度或隐藏背景真值恢复。
3. **原皮肤覆盖截断了积分路径。** 独立审查在 hair 的真实 neck ROI 发现逆映射平台和反向折叠。速度延伸后，该 ROI 的最小离散 Jacobian 从约 -0.359 提升到 0.825；近塌缩计数 680→0，负行列式计数 219→0。此处只描述同一 ROI 的场测量，不代表整图或所有幅度无折叠。
4. **恢复运动后暴露静态控制与源层冲突。** halo 与旧静态 pins 同用时出现梳齿；去掉相应 pins 的消融消除了梳齿，但仍有色块。因此不能把取消约束当作最终修复。
5. **候选模式名单漏接既有材质路径。** 全局代码复核确认新模式绕过了已接受的 shared hair promotion 路径。C 版本恢复三个相关模式名单，makeup 上侧边缘改善，hair 耳旁仍失败。D 恢复已有 connected-ear 模式，但旧 hair 输入的 `ears/accessories` 全为零，路径未启用，输出 SHA 与 C 相同；不能把无变化记成通过。E 恢复 `attachment_fixed` 连续障碍后耳齿改善，但旧颈残线仍在。
6. **源颜色证据被 alpha 零值提前拒绝。** `scene_edge` 在独立 `portrait_support` 已提供人物支持时，允许 alpha 为零/极低的点使用已有 fallback 颜色证据。旧 hair 长颈残线显著减少；尾点 `(790,845)` 的 support 与 alpha 均为零，仍保留灰褐色残色。该修复没有把未知外景通用提升为前景，也不代表 matting 完成。
7. **曲线投影与实际求解的固定自由度不一致。** 投影原以权重 ≤0.001 固定，实际场以 ≤0.05 固定，使低增益点得到无法执行的曲线目标。现统一为 `MIN_CONTROL_GAIN=0.05`，fresh hair 右侧平台明显改善，左侧回折约 2px→1px；下颌控制点实际仍为 -8.987px/+8.928px，未通过整体降低形变换取改善。
8. **胡须源 alpha 偏低、残差采样与最终合成使用了不同背景。** 同源完整头部 MEMatte alpha 消除了男士旧胡须残边，但仅改 alpha 仍留下左耳颈浅色条。继续固定同一神经 paired B，使源层分解、残差采样与最终合成共享底图；显式源纹理反例原 159 个检查点全部错误、最大 44 RGB 级，修复后全部正确、最大误差 0。零值/微小参数及保护区回归通过。两图实片结果见下，不外推未知背景真值。

## 当前证据

- `face::joint_domain::tests`：**9 passed / 0 failed**。包含真实 `VelocityGrid::transport` 的正负行程、逆向恢复、32 个方向、大坐标、远处背景/裁剪边界不动、硬障碍不可越过及无源域弃权。
- `scene_edge` 模块 **18 项通过**，相应实际 trace 回归 **1 项通过**。固定 paired B 阶段共享 `face::` 定向回归为 **225 passed / 0 failed / 3 ignored**，日志 `/tmp/face-replay-20261117/paired-scene-final-regressions.log`；源证据阶段 224/0/3、halo 阶段 222/0/3 分别保留在 `source-evidence-regressions.log`、`halo-regressions.log`。3 项 ignored 未执行，不能算通过。早期新 halo 的红测是新行为尚未实现，不代表旧回归已失败；旧跨零反例确有 4 RGB 级跳变，现已单独修复。
- makeup、hair 的 `faceWidth=±60` 已做分阶段实图对照。旧 hair 与 fresh hair 不能合并判定：fresh 使用同一照片的 784×1024 规范化源及最新检测，`ears` 非空，明显裂口/残线消失，但初始 ±60 仍有曲线平台和左台阶；统一固定自由度后有进一步改善，左接点仍需验收。
- 同源完整神经 alpha + 固定 paired B 的两图四结果已独立审查：`holdout-2379004 ±60` 所审旧胡须线、耳颈白边/浅色条与蓝衣污染未再出现；fresh hair `±60` 未见新增颜色缝，`+60` 左脸颈约 1px 几何小回折仍在。仅男士上述缺陷通过，不将两图都记为完整自然度通过。
- user、woman、dark 和 Lupita 的正确编码新回放仍在审查，不能预填通过；旧错误编码回放不进入画质判断，见下一节。
- 初版用户图 `faceWidth=60` 的 changedPixels 从 12,771 降到 9,934、最大 RGB 差从 157 降到 39，只是历史响应压小的证据，不作画质改善结论。

当前不能称通用可用或 8 分；限定场景最新通过范围见页首。旧源自动路径的残色、fresh hair 左接点和更多姿态泛化仍待验收；应在相同源层和实际请求下继续比较，不能把更新检测证据和算法变化混作单变量因果。完整门槛仍是 user、makeup、dark、woman、tilted 的 `0/±30/±60` 原尺寸双向验收，核对轮廓响应、连续性、五官与背景保护。无响应、靠降幅度或跳过失败像素换取“无裂缝”均不通过。

## 扩样输入编码错误与纠正

三图及 Lupita 准备脚本曾把 `physicalAlpha.data` 直接写成 `[0,1]`，而 `PreparedMask` 的实际 raster 合同是 JSON 数值转 f32 再除以 255，导致 renderer 收到的 alpha 被额外缩小 255 倍。此前只验证原始 numpy 层回裁，遗漏了序列化后的解码合同；因此旧 `neural-source-pair-more/extra` 成片是无效输入实验，不能据其浮雕/残影判定算法画质或回归。

两脚本和四图数据现改为**浮点 `alpha×255`，不 round、不量化为 u8**。新验证读取实际导出的 JSON，按 renderer 的 `f32/255` 解码，再回裁完整光学域和 replayView 对比源 cache。四图最大绝对误差均为 `5.960464477539063e-08`（容差 `1e-7`），零/一端点和 0.5 支持阈值完全保留；该检查验证输入传递，不是照片效果通过。

新输出目录为 `neural-source-pair-more-alpha255/` 和 `neural-source-pair-extra-alpha255/`，均关闭 trace。旧输出保留并加 `INVALID-INPUT.json`，错误 alpha、spec、provenance 与脚本归档在 `invalid-alpha01-inputs/`。正确输入的多图结果仍待审查。

## 证据位置

本轮简要审计见 [`face-width-continuity-20261117.md`](../qa/face-width-continuity-20261117.md)，逐点实片复核见 [`face-fresh-source-shape-audit-20261117.md`](../qa/face-fresh-source-shape-audit-20261117.md)，胡须与背景分层诊断见 [`face-beard-source-layer-audit-20261117.md`](../qa/face-beard-source-layer-audit-20261117.md)。回放输出、流场 trace 和逐版本 spec 在 `/tmp/face-replay-20261117/`，源图路径记录在 spec，没有复制进文档目录：

- `neck-write-woman/`、`neck-write-rest/`、`neck-write-hair-final/`：删除脸/颈硬写入门槛后的真实回放。
- `halo-makeup-hair/`：速度延伸与既有静态 pins 的成片及流场。
- `halo-layerpins-makeup/`：静态 pins 消融。
- `halo-shared-makeup-hair/`：C，恢复既有 shared hair promotion 路径；每个目录均含 `metrics.json`、成片和 `*.trace/`。
- `halo-shared-ear-hair/`、`halo-shared-barrier-hair/`：D、E 的旧源回放；不能与 fresh-source 混记。
- `halo-fresh-source-hair/`：784×1024 最新检测源，耳材料非空。
- `halo-scene-evidence-hair/`：源颜色证据修复；`halo-consistent-dof-hair/`：在该版本基础上只统一固定自由度阈值。
- `consistent-dof-multiphoto/`：较早 5 原照多强度回放，男士源层反例由此继续定位，不是当前神经配对版本验收。
- `neural-source-pair-comparison/`：男士与 fresh hair 两图四结果；局部对照 `/tmp/neural-pair-neck-before-after.png`、`/tmp/neural-pair-fresh-left-three.png`。
- `neural-pair-more-spec.json`、`neural-pair-extra-spec.json`：正确 alpha255 输入；对应 `*-provenance.json` 的 `physicalAlphaRendererDecodeAudit` 记录解码再回裁检查，结果待审。
- `hair-before-halo-sampling-metrics.json`、`hair-after-halo-sampling-metrics.json`：旧 hair 全图绝对 ROI `[760,820)×[800,930)`；只统计非 frozen、source alpha≥0.05、inverse 位移≥0.5px 的位置。中心差分计算 inverse Jacobian，`det<0.05` 定义近塌缩、`det<0` 定义反向；修复前后分别为 4612/5015 个有效点。脚本为 `/tmp/face_sampling_boundary_metrics.py`。

这些 `/tmp` 路径是本次本地证据，可能随临时目录清理失效，不是已提交的测试数据或可移植照片基准。早期失败的详细记录见 [`face-width-motion-domain-causal-20261117.md`](../qa/face-width-motion-domain-causal-20261117.md)，其旧测试数量与跨零失败状态由本页最新结果补充，不倒改历史证据。

## 验证

```sh
cargo test --manifest-path core/image-core/Cargo.toml --lib face::joint_domain::tests -- --nocapture
```

单图研究回放使用 `FACE_PROFILE_REPLAY_SPEC` 指向临时 spec；大图输出应写入 `/tmp`，不提交照片和生成物。

## 完整源资料回放与计算优化（本轮续）

- `scripts/prepare_face_source_replay.py` 将同源缓存导出为可复用实验输入，显式校验原 PNG SHA、完整解码 RGBA、graph 绑定、观察区域、alpha/B 和实际 JSON 解码回裁。alpha 缓存为 `[0,1]` 的 f32，raster 接口为 `[0,255]` 浮点数；不量化为 u8。14 项适配测试通过，包含此前 opaque=1 被错读为 1/255 的反例。
- public graph 与 cache graph 的两个已支持私有源资料分开传递：`sourceMotionInterface` 和 `sourceMovingAttachmentSupport`，默认保留原值。显式 `--omit-source-permissions` 才是省略消融，provenance 记录 applied/omitted。未知私有参数拒绝，不能悄悄丢掉来源合同。
- Lupita −60 的耳下 S 形缺陷经消融确认：只恢复 interface 不消失；恢复 attachment support 后消失，attachment-only 与 both 输出逐字节一致。随后完整源资料的 ±30/±60 未复发。该计划只提供移动许可，**不包含饰品刚性或旧离线求解器 W10 材料约束**。
- `PositiveKernel` 按半径单元索引相邻控制，保留原控制顺序及 f64 累加公式。极端坐标/非有限查询退回原稠密计算。9 项模块测试通过，包含独立稠密参考的逐位比较；共享人脸回归为 228 passed / 0 failed / 3 ignored。user 与完整源资料 Lupita 各 ±60 的四幅结果与索引前逐像素一致。
- 独立单线程 native kernel 基准使用 Lupita 的 426 个控制和 5 轮真实控制值，311,375 次查询/轮、4 轮交替：代表半径 32/64/96 的稠密→索引中位耗时分别为 162.23→27.59、195.00→74.85、242.67→148.55ms。各轮 checksum 相同；机器存在后台负载，半径是代表设置，**这些不是整图、WebAssembly 或实时滑块性能**。本地完整记录 `/tmp/positive-kernel-bench-20261117/summary.json`。

原生研究入口仍是 `cfg(test)`。源 alpha 与隐藏背景由模型估计，不是真值；适配正确、计算等价与定向实片通过不构成所有人物/强度/功能的普遍保证。后续继续做独立控制与身形真实坏例，不用执行成功率替代效果。
