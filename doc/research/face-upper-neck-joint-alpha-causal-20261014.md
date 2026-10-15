# face + upper-neck 联合前景的只读因果分析

日期：2026-10-14。分析基于生产 v11 和隔离 v17/v18，未修改生产。v19 若验证，须单独归档一次假设，不能继承 v18 硬neck pin。

## 因果结论

可以验证这条路线：同一连续、可逆 SVF 对真实连通 face+neck 物理前景运输，连通源域的像在数学上仍连通。当前下颌面具被当作独立前景、neck作为静止underlayer，内部face/neck语义交界被误当作需要补全的透明边界；共同运输可取消这处人为切口，减少把hidden neck猜成scene和旧阴影残留。仍会在真正皮肤/背景边缘产生新露出场景，不会消除遮挡补全需求。二维RGB跟随也不能保证真实三维光照。

不能保证20px：耳/饰品/user保护与目标距离不足时仍有真实可行性冲突。更重要的是faceContour的一部分只是内部face/neck标签边界，搬动标签20px不等于最终外部skin silhouette移动20px。须分别测face标签/真实联合外轮廓/原尺寸接缝。v18表明将neck端点固定可把+100实效从20px压到8px；joint实验应保留v17 face目标，不再硬固定上颈连接。

## 输入能否支持

现有 parser 的 `skin` 是原始分类组 `[1,10,14]`（脸skin、nose、neck）概率相加后统一阈值化，跨face/neck交界比“两个分别阈值后的mask相加”更可靠；但它不包括眉眼/嘴内部，不能单独作完整人脸前景。`max(faceContourMask, skin)` 仅可作为有限的启发trimap种子，再用原RGB `refine_portrait_alpha` 估计物理边界；它不是raw完整类别union或物理真值。两个已阈值化group在互斥 mouth11=.45 / neck14=.45 / bg=.1时 max=.15625，而raw完整union=.9；当前face-skin1/neck14界面的改善依赖skin自身原始聚合，不能推广到所有材质交界。只采用与源脸邻接的可信neck组件，不能用全personAlpha（包含衣服、耳发和其他实例）。类别置信度不是amodal材质真值；不能凭neck bbox绘制隐藏skin。

`accessories` 当前只含 `[6,9]`，没有显式 necklace(15)/cloth(16) 图层。源neck外的高置信 person/context 固定提供保守保护，但这不等价于独立项链检测。实验须报告项链附近source像素固定情况，不能主动扩大协议假装已有necklace/collar语义。

## 当前代码阻碍与正确边界

| 位置（生产 face.rs） | 当前假设 | joint 分支需要 |
|---|---|---|
| 1500 known_layer / 1551 neck_lighting_known | neck是静止下层，face以外neck可做donor | 联合mask内skin归foreground，静止低颈仍同一物理skin，只是velocity=0 |
| 1570 moving_alpha | 只采face_membership | 可信face+neck联合物理alpha，head保留背景上下文，耳发不纳入moving |
| 1584 independent_head cap | head subject=0强制alpha=0 | 取消对可信neck的头部裁切，保留独立固定层/选区约束 |
| 1634 old_coverage skip | 收缩只编辑旧face内 | 以联合物理alpha定义旧coverage；真正scene外保持exact |
| 1662 clip_supported_jaw_opacity | 对独立face的新下巴alpha再裁切 | joint停用，不能制造第二个face内部透明切口 |
| 1689 restore_neck_lighting | 静止neck纹理 + 单独移动光照 | joint停用，已随flow运输的neck阴影不能再次搬动 |
| reconstruct（402起） | 对face holes传播nearest scene/neck/hair下层 | holes仅源联合前景新露出的真实underlayer；不再补内部face/neck语义边界 |
| region/mask/protect/frozen | 用户授权区域和显式固定精确 | 共用velocity gate，区域外identity；若实验范围扩大，graph明确声明并逐像素报告 |

联合物理alpha应包含连续可见neck，运动只作用上颈；不能在任意“upper-neck截止线”把alpha切成0。下颈/collar通过velocity连续归零固定。脸尺度继续取landmarks，不能把联合skin bbox当作face Frame。union外轮廓、face内部标签和固定层分别维护，不共享错误的头部ellipse。

## 测试契约

现有函数单测 `uncovered_neck_uses_observed_shadow_detail_and_never_changes_existing_neck`、`contracted_jaw_repositions_neck_illumination_without_warping_texture_or_scene` 是旧静止neck函数的有效单测，可保留用于旧分支，不能用它们否定有意运输上颈的新分支，也不能删除它们掩盖退步。end-to-end `jaw_reveals_neck...` 中低颈214..238的exact要求继续适用，但真实外轮廓需跟联合alpha测量，不能仅用face颜色阈值。

`review_face_refinement.py` 的 head-free/no-neck>0.8 exact限制会把可信混合上颈误计为越界；joint应单独声明允许的neck运动mask，同时保留scene、耳发饰品、另一个人、下颈、user protection/ROI精确不变。禁止扩大允许范围只为让检查通过。

一次候选的验收最少包括原照同detector四档 ±30/±100、明确input/source/binary/output哈希、真实face与union边缘位移、零内部scene holes、颈部纹理/阴影连贯、下颈/collar固定、无fold；视觉由原尺寸对比核对。若从现有skin/neck/RGB无法得到可信连通matte，应停止并记录，不画假的neck边界。
