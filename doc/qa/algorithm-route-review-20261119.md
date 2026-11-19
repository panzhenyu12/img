# 人像形变算法路线复审（2026-11-19）

## 结论

当前整套人像重塑仍**不可用，未达到 8/10**。局部实验可以证明“给定正确的源身份、光学覆盖和遮挡顺序后，连续场能产生有限响应”，但不能替代自动照片上的完整身份识别、边界 matte 和自然度验收。

## 身体路线的硬阻断

`core/image-core/src/body.rs` 的普通 `separated_adjust` 入口中，`geometry_target()` 将
`subjectMask ∩ partMask` 经过 `refine_alpha` 得到的结果同时用于：

1. 运动目标和截面测量；
2. `body_layer_colors(source, target)` 的前景/背景分解。

这把“哪种材料有运动权限”和“该像素有多少物理前景覆盖”混成一个量。服装、手、包、
外套相邻或互相遮挡时，即使数学场连续，也会搬动错误材料或用错误 donor 填空。继续调
阈值、半径或尾部宽度不能消除身份信息缺失。

`body_source_layers::SourceLayers` 已经把 `optical_alpha` 与 motion field 分开，是正确
的低层方向，但目前只有一个 moving layer，仍未表达衣身、前袖、手、外套和固定场景的
可见遮挡图。因此它只能作为受控 source-layer 实验接口，不能直接视为通用收腰功能。

## 推荐的最小落地顺序

1. 由 PartGLEE/SAM/人工确认**源身份**，只生成 motion `partMask` 和实例关系；不把它
   当作 optical alpha。
2. 在每个源身份域内独立估计同源 `optical_alpha`、前景颜色和固定背景，保留 trimap
   unknown；配对失败或支持不完整时拒绝该局部编辑。
3. 按可见遮挡顺序拆成多个 moving/fixed layer，使用 `SourceLayers` 运输物理覆盖，
   露出的孔洞交给受限的局部修补；隐藏身体形状不得由单层 matte 猜出来。
4. 先在新的可观测腰/袖/腿照片上验收 `0/±30`，再评估更强档位和泛化。

## 脸部路线的硬阻断

生产 `face.rs` 的 `target_contour()` 仍以固定解剖坐标的 y profile 和横向 across 投影
生成目标；近期已把 lateral hard clamp 改成平滑饱和，并把 `ContourGrid` 的窄边带切换
改成连续混合。这些改动只消除一个数值不连续来源，没有改变目标曲线的解剖先验或源材质
归属。生产 Web 路径只把 parser 的 semantic masks/subject mask 交给 WASM，没有把
`FaceSourceLayers` 的物理 alpha/F/B 合同带入。因此脸宽、颧骨和下颌在复杂发丝、耳朵、
颈部边界上仍可能出现裂缝、双边或不自然曲线；本轮尚无真实照片成片证据证明该补丁已解决。

当前 `adjust_impl` 的 upper-face adaptive 分支虽使用了 `.unwrap_or(true)` 的默认尝试，
但没有 `FaceSourceLayers` 时 `promote_with_source` 只能读取 parser 的
`faceContourMask`、`hair`、`skin` 语义栅格；这不是物理 alpha/F/B。`faceWidth` 的
shared-cheek 入口还要求 `source_layers.is_some()`。因此“默认分支存在”不能视为 Web
端已经接入源材质证据或完成效果验收。

脸部下一项应先接通同源 physical contour alpha/F/B，并让 semantic contour 只控制 motion
geometry；缺少源证据时拒绝或回退手工轮廓。随后再将闭环源弧长/曲率约束的 topology fairing
接入目标曲线，并用连续的边缘运输替换窄硬切换。没有真实照片 100% crop 的视觉通过前，
不应以单测数量或低强度响应宣布可用。

## 当前证据口径

官方 PartGLEE q104 在 `side-fashion` 的 `±30` 仅是有限 source-layer 对照：可见丹宁
边有双向连续响应，前景袖/手/包/外套未见新增变化，但其余角色仍手工冻结，前景 matte
保留 10 px unknown 和 16 px 封闭漏检。生成报告的 `visualAccepted=false`/
`qualityAccepted=false` 继续有效。该证据不能计入整体可用或 8 分。
