# 身体部位 V3 实拍逐项复核

日期：2026-10-13。五张相同实拍、15项单部位、参数60、最长边768px。**实际轮廓和部位保护有净改善，但当前15项都未达到每功能8分目标，不宣称身形效果完成。**

使用复制后保持不可变的 native renderer `f315df824db93bd592b159b73b2d5988fa94f05f8cd1a8fcedd3f96ae06f558d`，每项执行前后核hash。来源图、parser cache、graph与output的SHA及固定副本路径见 [provenance](body-parts-v3-provenance.json) 和 [metrics](body-parts-v3-metrics.json)。V2原产物未重算；所有V3裁切已按原图/V2/V3/差分x8逐张目视。

硬条件：真正轮廓改变、背景固定、非目标部位保护。效果/自然度/部位保护/编辑效率权重35/35/20/10，每维至少7且总分至少8才通过。这是当前样片的主观工程评分，不是算法在所有照片上的统计质量。

15项都有非零编辑；所提供保护mask、非目标mask、原已知背景mask与support外变化均为0。**这些是按输入蒙版判定的协议指标，不能证明所有真实非目标对象都被分区正确，也不自动代表自然。**同类衣物误归入torso时仍会错误形变。

本组native CLI总耗时1969–5803ms（读取与PNG输出也在内），Pose+部位层构建1436–9501ms；运行时存在并行开发负载，不能据此给实时/峰值内存结论。编辑效率统一暂记6。模型推理不包含在该编辑计时内；独立Worker真实冷解析约26.02秒，见 [runtime](body-parser-browser-runtime.json)。

|实拍/部位|效果/自然度/保护/效率|总分|目视结论|
|---|---|---|---|
|[fashion-standing / waist](body-parts-v3/fashion-standing-waist.png)|8/7/8/6|7.45|衬衫两侧真实收缩，腰带、中轴和手保持；袖子交界仍有薄色边，尚不足自然度8。|
|[fashion-standing / left-upper-arm](body-parts-v3/fashion-standing-left-upper-arm.png)|8/7/8/6|7.45|袖子外侧真实内移，布纹保持；比V2浅色重边减少，肩/袖的极细边仍可见。|
|[fashion-standing / left-thigh](body-parts-v3/fashion-standing-left-thigh.png)|8/7/8/6|7.45|裤腿两侧收窄，腿间旧阶梯灰残影显著减少；植物附近窄带仍有软延拓，不能保证原结构恢复。|
|[fashion-standing / right-upper-arm](body-parts-v3/fashion-standing-right-upper-arm.png)|8/6/8/6|7.10|弯臂上袖真实收窄、前臂固定。4×复核确认明显亮细线原图已有且保持原位置，不能计为新增鬼影；新轮廓转折/窄带仍需独立层诊断，本轮自然度评分暂保守保留。|
|[fashion-standing / right-thigh](body-parts-v3/fashion-standing-right-thigh.png)|8/7/8/6|7.45|裤腿真实内移、腿间旧阶梯减少；地面窄带轻微模糊与色缝仍可见。|
|[fullbody-skirt / waist](body-parts-v3/fullbody-skirt-waist.png)|7/7/8/6|7.10|已包含真实裙腰，腰侧和裙腰轮廓有变化，袖子保持；可见腰被宽衣/袖遮挡，效果不宜夸大。|
|[fullbody-skirt / left-upper-arm](body-parts-v3/fullbody-skirt-left-upper-arm.png)|7/7/8/6|7.10|浅色袖子真实外侧略收窄、头发保持；袖边的细暗缝仍可见，隐藏肢体没有重建。|
|[fullbody-skirt / right-upper-arm](body-parts-v3/fullbody-skirt-right-upper-arm.png)|7/7/8/6|7.10|袖子外侧收窄、蕾丝大致保持；袖子与衣身交接细边仍不足以给8分。|
|[fullbody-side-fashion / waist](body-parts-v3/fullbody-side-fashion-waist.png)|7/6/7/6|6.55|包和手保持、裤腰真实改变，旧外套鬼影减少；垂挂外套与上衣同类，仍不能保证准确独立归属。|
|[fullbody-side-fashion / right-upper-arm](body-parts-v3/fullbody-side-fashion-right-upper-arm.png)|8/7/8/6|7.45|粉袖可见轮廓真实收窄，V2手/头发重影消除；局部袖边仍有细色边。|
|[tilted-shirt / waist](body-parts-v3/tilted-shirt-waist.png)|6/6/6/6|6.00|右侧宽衣真实收缩；叠在腰前的同类宽袖仍可能分给torso，局部衣料归属和边缘自然度均失败。|
|[tilted-shirt / left-upper-arm](body-parts-v3/tilted-shirt-left-upper-arm.png)|8/7/8/6|7.45|宽袖外轮廓确实改变、头发固定；袖口转折与低对比背景仍有细边。|
|[action-suit / waist](body-parts-v3/action-suit-waist.png)|7/5/8/6|6.40|电脑主体和边框保持，外套腰侧收缩；白衬衫/外套内部边出现小硬台阶，判自然度失败。|
|[action-suit / left-upper-arm](body-parts-v3/action-suit-left-upper-arm.png)|7/6/8/6|6.75|抬臂上下边略收窄、躯干大致固定；肩接缝/下袖边有硬转折，效果和自然度未达标。|
|[action-suit / right-upper-arm](body-parts-v3/action-suit-right-upper-arm.png)|8/7/8/6|7.45|上臂外轮廓真实内移、前臂与电脑保持；腋下/袖边窄带仍可见。|

## 净改善与剩余失败

FASHN真实像素+Pose竞争骨轴得到actual partMask，手/脸/头发/包成为固定层；真实非对称宽袖不再被近侧启发式剪细。侧身上臂的手/头发鬼影、裤腿间旧阶梯残影、西装电脑误动比V2明显减少。新露出coverage判定和固定遮挡围住的安全供体路径使15项均可正常编辑。

剩余失败集中于同类衣物竞争分区（叠袖、垂挂外套），内部层交界硬台阶，以及复杂背景窄带细边。颜色线alpha细化只能改过渡边，不能恢复错误语义；diffusion不恢复真实建筑/植物结构。保持这批失败作为下一轮基线，不能靠减小强度或冻结整个部位达标。

## 验证范围与复现限制

这组编辑使用固定官方Torch缓存概率，后处理为logits先全图bilinear、softmax再缩概率；Web当前先将logits重采样到最长边384再softmax，非逐数值一致。真正浏览器Worker已另外运行同ONNX，Torch/ORT同输入parity和Web/OpenCV输入rounding误差均已保存；不能以缓存编辑代替产品自动检测的逐图端到端验收。

21项body release测试和2项frontend body-parts测试通过。native/WASM统一构建、热缓存复用、不同source/part失效由主任务单独验证；本报告仅包含native固定副本。肩/腹/胯/前臂/小腿/全身组合没有本组逐功能实拍评分，不从本15项推断全部功能。

复现：先 `make body-models` 准备模型，启动Web；使用 `BODY_RENDER_BINARY=/绝对路径/不可变副本 node scripts/replay_body_parts.mjs` 重放。必须保留本轮cache与V2原产物；视觉评分需重新目视，不由脚本按差分提高。数据见 [review JSON](body-parts-v3-review.json)，实现见 [FUNC-012](../func/FUNC-012-body-semantic-masks.md)。

复核纠正：4×原图/输出裁切确认右上臂旁的明显亮细线为原图已有结构，保持原位置；不能以这条线证明新生成鬼影，也不应为了抹除它去改固定背景。自然度分暂保守保留，新边缘归因另做层诊断。
