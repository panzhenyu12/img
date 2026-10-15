# 整眼轨迹核心 V5 首版：失败归档

2026-10-14。生产未变，方向只涉及 eyeTilt / eyeSpacing。Root 与子代理独立查看原尺寸照片，都拒绝本轮审美：+100眼距鼻梁顶端出现双侧三角亮块与新折痕，倾斜虽眼球姿态统一，双眼皮过渡有捏痕。

当前代码保留原眼polygon的SE(2)扫掠凸包作为权重1核心，外部以到核心距离的smoothstep过渡。原眉、另一只眼与核心的距离限制过渡宽度；gap<=0时整warp标blocked。

## 数值成功的有限范围

完整眼缘fixture两项×四档×双眼的边长相对误差<.0001，actual最终inverse mapper回原点误差<.00011px。1024用户原照eyeTilt+100整眼面积比1.0000009/1.0000035，边长相对误差<.000042；eyeSpacing+100面积/边长一致，解析刚性位置误差<.00085px。内部虹膜指标仅为iris-sized circle几何代理；实际虹膜与眼睑仍需照片核对。

这证明核心的整眼刚性方向成立，不能证明外部皮肤自然或全部组合可用。[照片与完整指标](../qa/photo-report-selected.json)来自immutable初版test worker的本地feature pass，尚未运行public最终opacity/mask composite；本轮选区在目标眼域为完整覆盖。初版和追加审计版源码分别保存，不混用其binary/hash。

![倾斜三列，原图/V4b/V5](../qa/user-eyeTilt100-comparison.png)

![眼距三列，原图/V4b/V5](../qa/user-eyeSpacing100-comparison.png)

## 失败回归

保留V4b全部31项测试，再加入新整眼一项，结果31pass/1fail/3ignored。失败是原`detected_brow_final_mapping_stays_invertible_at_extremes_and_boundaries`里的browPeak=-100、browTail=-100、eyeSpacing=-100：移动后眼包络额外受到眉场约.30049px位移，超过原.05px上限。此被检查场景fold=0、minJacobian=.11937、reverse error=.001587、fixed donor=0；测试在首个失败停止，不能声称所有86场景通过。

专门的硬block负例跑actual最终mapper及RGBA。约束阈值87.488708..87.488716，滑杆对87.488609..87.488815导致最大映射跳6.368087px、最大RGBA跳245。见[阈值记录](../qa/first-hard-block-continuity.json)。该测试故意失败并保留断言；没有删除断言或将其当通过。

## 后续单一方向

继续“整眼刚性、皮肤延拓”这一方向；采用包含可见眼睑/眼袋邻接范围的核心，外部研究screened biharmonic速度延拓。眉/鼻梁/非目标五官是独立固定边界；同时有眉调整时，必须使用移动眼扫掠域保护，不能只用静态eye bounds。几何冲突须沿请求轨迹连续限制可行姿态、明确记录requested/applied，不再整项骤然关闭。

扩大包络仍只是现有眼检测加邻接估计，不等价于独立检测了双眼皮和眼袋真值。后续先以这张实际失败照片的原尺寸±100验证，不继续为已经失败的首版增加批量照片。

代码、两个worker与证据哈希见[provenance.json](provenance.json)。初版失败不接生产、不宣称8分。
