# 身体部位 V4：遮挡截面连续性复核

日期：2026-10-13。相同五张实拍、15项单部位、相同graph/参数60、最长边768px；逐图查看原图/V3/V4/差分x8。**V4去掉了西装腰部硬台阶，保持轮廓效果和固定上下文；当前仍0/15达到每功能8分目标。**

不可变renderer SHA256：`515ed394104c4b86d17501fbf21dde38c3b9e3bc354bbabefa8c3ef4c321e544`，每项前后校验，见 [provenance](body-parts-v4-provenance.json) / [metrics](body-parts-v4-metrics.json)。V3原产物和评分保留，见 [V3评审](body-parts-v3-review.md)。

## 此轮修复

V3骨轴经过固定电脑时，partMask在核心没有可测截面。相邻截面有任一半径为0时，旧逻辑将整截面位移立即归零，导致远离电脑的白衬衫边也突然回到原位置。V4按到缺失截面的轴向距离连续锚定位移，缺失区保持固定，其他有效截面仍使用相同幅度的真实轮廓映射。不是降低全局参数或冻结整块腰部。

新增回归要求：遮挡处位移为0；正常截面位移仍大于3px；逐像素轴向位移跳变小于1px。22项body release测试通过。实拍action-suit腰白衬衫硬台阶消失，电脑保持，腰外边仍内移；抬臂袖边转折也更连续。fashion-standing、fullbody-skirt、tilted-shirt的腰结果与V3 output SHA逐字节相同，局部改动没有削弱无该遮挡的腰部。

## 逐项评分

权重：效果/自然度/部位保护/编辑效率35/35/20/10；每维至少7、总分至少8才通过。主观工程评分仅针对这批样片，不能外推所有照片。V4仅将已目视改善的西装腰/抬臂自然度上调，其他评分保留，失败不隐藏。

|实拍/部位|效果/自然度/保护/效率|总分|目视结果|
|---|---|---|---|
|[fashion-standing / waist](body-parts-v4/fashion-standing-waist.png)|8/7/8/6|7.45|衬衫两侧真实收缩，腰带、中轴和手保持；袖子交界仍有薄色边，尚不足自然度8。|
|[fashion-standing / left-upper-arm](body-parts-v4/fashion-standing-left-upper-arm.png)|8/7/8/6|7.45|袖子外侧真实内移，布纹保持；比V2浅色重边减少，肩/袖的极细边仍可见。|
|[fashion-standing / left-thigh](body-parts-v4/fashion-standing-left-thigh.png)|8/7/8/6|7.45|裤腿两侧收窄，腿间旧阶梯灰残影显著减少；植物附近窄带仍有软延拓，不能保证原结构恢复。|
|[fashion-standing / right-upper-arm](body-parts-v4/fashion-standing-right-upper-arm.png)|8/6/8/6|7.10|弯臂上袖真实收窄、前臂固定。4×复核确认明显亮细线原图已有且保持原位置，不能计为新增鬼影；新轮廓转折/窄带仍需独立层诊断，本轮自然度评分暂保守保留。|
|[fashion-standing / right-thigh](body-parts-v4/fashion-standing-right-thigh.png)|8/7/8/6|7.45|裤腿真实内移、腿间旧阶梯减少；地面窄带轻微模糊与色缝仍可见。|
|[fullbody-skirt / waist](body-parts-v4/fullbody-skirt-waist.png)|7/7/8/6|7.10|已包含真实裙腰，腰侧和裙腰轮廓有变化，袖子保持；可见腰被宽衣/袖遮挡，效果不宜夸大。|
|[fullbody-skirt / left-upper-arm](body-parts-v4/fullbody-skirt-left-upper-arm.png)|7/7/8/6|7.10|浅色袖子真实外侧略收窄、头发保持；袖边的细暗缝仍可见，隐藏肢体没有重建。|
|[fullbody-skirt / right-upper-arm](body-parts-v4/fullbody-skirt-right-upper-arm.png)|7/7/8/6|7.10|袖子外侧收窄、蕾丝大致保持；袖子与衣身交接细边仍不足以给8分。|
|[fullbody-side-fashion / waist](body-parts-v4/fullbody-side-fashion-waist.png)|7/6/7/6|6.55|包和手保持、裤腰真实改变，旧外套鬼影减少；垂挂外套与上衣同类，仍不能保证准确独立归属。|
|[fullbody-side-fashion / right-upper-arm](body-parts-v4/fullbody-side-fashion-right-upper-arm.png)|8/7/8/6|7.45|粉袖可见轮廓真实收窄，V2手/头发重影消除；局部袖边仍有细色边。|
|[tilted-shirt / waist](body-parts-v4/tilted-shirt-waist.png)|6/6/6/6|6.00|右侧宽衣真实收缩；叠在腰前的同类宽袖仍可能分给torso，局部衣料归属和边缘自然度均失败。|
|[tilted-shirt / left-upper-arm](body-parts-v4/tilted-shirt-left-upper-arm.png)|8/7/8/6|7.45|宽袖外轮廓确实改变、头发固定；袖口转折与低对比背景仍有细边。|
|[action-suit / waist](body-parts-v4/action-suit-waist.png)|7/7/8/6|7.10|电脑主体/边框保持、外套可见腰侧真实收缩；V3白衬衫硬台阶已消失，交界连续。细色边和局部衣纹仍未达到自然度8。|
|[action-suit / left-upper-arm](body-parts-v4/action-suit-left-upper-arm.png)|7/7/8/6|7.10|抬臂上下边仍可见收窄；V3肩侧/下袖边的硬转折减少，遮挡截面附近连续固定。轮廓细边与效果幅度仍未达8。|
|[action-suit / right-upper-arm](body-parts-v4/action-suit-right-upper-arm.png)|8/7/8/6|7.45|上臂外轮廓真实内移、前臂与电脑保持；腋下/袖边窄带仍可见。|

## 未通过项与验证限制

倾斜宽衣在腰前的叠袖、侧身垂挂外套仍可能错误归入torso；同类衣物分区不是真值。细边、复杂背景窄带的结构延拓仍未达自然度8。身高/腿长/体态未实现专用算法；肩/腹/胯/前臂/小腿/全身组合未在本组逐功能目视验收。

四类contract指标（所提供保护mask、非目标mask、原已知背景mask和support外）变化均0，所有15项有非零编辑。它们不证明所有真实非目标对象都被分区正确，也不自动证明自然度。native CLI本轮2064–13075ms，Pose+layer 1638–12607ms，存在明显并行负载变化；编辑效率沿用暂记6，不据此宣称实时。

本组仍采用官方Torch缓存概率；Web当前input整数rounding/decoder略有不同，真实侧身Web与cache argmax差54/98304像素，见 [实际差异](body-parser-browser-cache-difference.json)。另行完成真实Worker/ORT推理和Torch/ORT同输入parity，并不替代五图产品自动检测的端到端验收。native/WASM和热缓存一致性由统一构建另验。

复现：`BODY_RENDER_BINARY=/绝对路径/不可变V4副本 BODY_REVIEW_VERSION=v4 BODY_PREVIOUS_VERSION=v3 node scripts/replay_body_parts.mjs`；默认仍可生成V3/V2。视觉分数需重新逐图审查。见 [review JSON](body-parts-v4-review.json) 与 [FUNC-012](../func/FUNC-012-body-semantic-masks.md)。

复核纠正：4×原图/输出裁切确认右上臂旁的明显亮细线为原图已有结构，保持原位置；不能以这条线证明新生成鬼影，也不应为了抹除它去改固定背景。自然度分暂保守保留，新边缘归因另做层诊断。
