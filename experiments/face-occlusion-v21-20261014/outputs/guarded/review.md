# 静止饰品v21：因果成立，当前效果仍拒绝

2026-10-14。只回放用户原图1024² +30/+100；原检测缓存不变，未改正式核心、公共WASM、前端或协议，未推送。source、graph、native binary与输出SHA和immutable路径见 `report.json`。首版、失败供体测试和源码另完整保留于 `../initial` / `../../src-field-only-first`。

## 解剖运动与遮挡职责

旧hair/ears/accessories全部参加约43px的解剖连接taper。耳环悬垂部分不是皮肤连接；v21保持它的最终原RGBA固定，并从motion attachment种子中分出，仅hair/ears保留连接距离。饰品仍留在原alpha/owned障碍且不落入context 4px锁域。打开旧attachment和旧donor两项native私有对照后，两档与当前正式v19全RGBA完全相同。

旧v19左y454.8/y482.7/y492侧控制的+100请求20.02px、preQP与实际均0。新运动语义下preQP均20.02px，实际14.4/18.6/19.6px；最大实际20.14px。+30最大实际6.009px。总幅度没有削低，上颊现在真正参与脸宽调整。

但+100耳环旁仍出现竖皮肤条，上颈仍继承脸宽运动。原尺寸独立审阅拒绝本候选，不能将上脸终于有移动称为8分算法。root另对源ZIM物理饰品alpha与face/skin支持作独立诊断；本报告不把它混入当前geometry变量。

## 全流程供体隔离

首版虽然最终饰品像素精确固定，单独改饰品RGB为绿/品红会影响外部800–1558像素、通道最大差138–143，证明旧reconstruct donor_frozen不保护alpha颜色校正、F/B估计与直接前景取样的全部通道。

guard阶段给内部颜色工作层延拓干净观测颜色，再用于alpha校正、F/B及前景采样，并强制hair重叠饰品也禁止reconstruct供体。它不改变原canonical或原graph，但影响344个source physical alpha像素，其中247在三类固定mask外（x369..643/y427..495），最大差.996。不能把它当作完全冻结物理alpha的单变量几何实验。

因此另渲染同guard的all-attachment v19对照，其source physical alpha与candidate全字节相同；它相对正式v19输出改变273/+30与324/+100像素，归因于供体隔离。完整对照在 `occlusion-contract`。guard后的+30/+100×绿/品红，饰品外变化全部0，输出硬饰品RGBA保持；单独验证的是coarse hard区域，未声称未知饰品/soft alpha全部被证明。

## 权限、运输与性能

两档耳/发/饰品、9实际五官内部、ROI外、原RGBA alpha、语义scene和y740之后低颈均0变化。+30/+100 source控制到达最大误差.0083/.2535px，采样Jacobian最低.558/.224、fold探针0。窄颈形状仍明显；低Jacobian虽未fold仍意味着强局部压缩，不能用数值通过替代审美。

两阶段的具体source请求/preQP/可行/实际、物理外皮肤alpha行宽/切线均在 `debug*-curves.json`。当前matte估计不是人工gold。native总耗时含CLI、JSON/PNG、matting与其他阶段；debug和并行编译使其不可当作稳定Web实时性能。无debug三次+100原尺寸重复及RGBA一致性另见 `quiet-repeat-report.json`。

默认完整release测试207 passed/3 ignored，integration10 passed；face::tests单独35 passed/1 ignored，新增了禁止饰品颜色/alpha影响工作层的检查；Clippy全target -D warnings通过。所有19项production来源SHA复核未变。单测、byte parity与硬权限通过不作为8分证据。

## 后续相近实际幅度的曲线目标

1. 先沿可信物理支持追真实可见外脸侧、源下颌弧与颈侧，分别记录；内部face/neck标签边界不冒充物理外轮廓，耳环遮挡缺段不当连接点。待root的物理支持首图通过后，冻结该同一source support再改目标。
2. 以可见源弧宽度比例保留个人左右骨相；使用非均匀间距积分的位移能量及source相对切线/曲率，保留合法转折。v20把+100压到12.5px的硬曲率上限不再沿用；用局部源形约束与明确可行上限，而不靠削弱总体幅度获得自然。
3. 在真正可编辑外脸侧选可信幅度锚点，同当前+30约6px、+100约20px达到位置比较；另报整个可见颊弧位移分布与下颌转角/颏弧，不能只以单个内部控制点最大值匹配。不可编辑耳根仍0，若目标因固定约束不可行应报告请求/可行差值。
4. 给真实颈侧独立source相对目标与允许位移预算，禁止从最近face标签目标直接复制20px；保留共同SVF/alpha及逐段连接。在上端可行目标及位移导数连续的前提下渐停到原低颈，核对颈宽、切线、局部应变与材质接缝。正负方向独立审核，不能硬钉一条横线制造材质切口。
5. 同幅度候选必须同时通过原尺寸审美、physical边缘/曲率及纹理/权限检查，然后才扩大两张原尺寸独立人像±30/±100回放与Web/native/Go parity。

这些是待验证的候选目标，未写成已达世界一流水平的结论。
