# FUNC-020 五官连续形变与眉毛眼睑保护

状态：V4b已实现并通过对应几何/采样回归。眉峰/眉尾连带眼睑、固定边界折叠、眼倾斜中心放大串扰与透明identity原字节丢失已修复；眉形自然度、其他五官的精修效果和实时性能尚未达到逐功能 8 分验收。

关联需求：BROW-001、BROW-002、BROW-003、BROW-004、BROW-006，以及已有眼睛、鼻子、嘴唇几何调节。眉毛倾斜、补缺和边缘材质重建未因本次改动新增完成。

## 用户行为

在“五官与妆容”导入图像、检测人脸后，通过滑杆调整眉长、眉宽、粗细、眉峰、眉尾和间距。眉毛形变应保留眼球、眼睑折线及眼线；眼睛同时调整时，眉毛不应额外扭曲对应眼睛。参数从 0 跨到小数值时，其他已开启的五官不应切换算法或突然变化。

## 前端实现

继续使用 MediaPipe 的语义 polygon 与当前图层输入坐标。Web 将同一 `feature.adjust` 操作图交给 Rust WASM；没有 AI Agent 时滑杆与导出可以独立使用。此次不改控件、模型定位或图层协议。此前检测上下文修复记录在 [FUNC-015](FUNC-015-skin-and-makeup-quality.md)。

## 后端实现

Go/Gin API 继续调用 Rust native `image-render`。形变数学与浏览器共用 `geometry.rs`，没有新增另一套 Go 像素算法或模型服务。

## Operation 与参数

Operation 仍为 `feature.adjust`。保留 `adjustments`、`semanticMasks`、`landmarks`、`region`、`mask`、`protectMask`、`opacity` 和 `blendMode`。AI/MCP 仍可调用同一参数；没有扩充接口。

```json
{"adjustments":{"browPeak":40,"browTail":20,"eyeSize":10}}
```

有语义定位的五官几何统一使用连续速度积分。颜色/妆容 pass、没有语义定位的手工模板、身体/衣物形变和 `face.rs` 的脸部轮廓路径不在这次改动范围内。

## 实际算法改变

原来把每个局部位移直接加到采样位置。眉毛支持域覆盖眼睛时会把眼睑一起弯曲；直接把位移乘眼睛保护权重又可能在保护过渡处折叠。V1、V3 失败补丁和诊断保留在 `experiments/brow-eye-damping-v1-rejected-20261014` 与 `experiments/brow-flow-v3-rejected-20261014`。

当前把局部缩放、平移和旋转的紧凑场作为 stationary velocity，用 RK4 求单位时间的流。检测的眼睛包围域沿眼线方向扩展到眼睑与眼线，眉毛速度在这个域内为零，并平滑过渡到周围皮肤。正负数值均按请求强度积分，没有通过降低最大滑杆或加模糊掩饰旧变形。

V4b旋转速度改为正确生成元`(-a*v,+a*u)`。旧有限旋转位移中的`cos(a)-1`径向项导致单独eyeTilt±100时眼中心forward局部面积额外放大约9.34%；现检测中心inverse Jacobian约0.999938、请求±0.3rad得到±0.300058rad。椭圆支持域的权重梯度仍使检测eye polygon forward顶点shoelace面积比0.980707–0.998189；中心稳定不等于整眼严格刚体或保面积。

共享remap对精确identity且在源图bounds内的位置直接复制RGBA，保留alpha=0的hidden RGB；真正移动的位置继续预乘alpha插值。公开graph核对identity最终RGBA，移动alpha变化属于合法重采样。浏览器Canvas解码PNG可能已丢hidden RGB，因此raw parity只验证共享核心输入输出，不把此修复描述为编辑器导入全程无损。

选区、保护、region 与 hair/ears/accessories 冻结像素先构成连续运动域：双线性插值权重、距零域 2px 的供体缓冲和 2–10px 的距离过渡都参与每个 RK4 stage。这样固定像素和硬边界已经位于零速度域内，SVF 路径直接返回积分后的坐标，避免最后把不允许的供体采样突然改成 identity。与眉毛组合的鼻矫正/二维对称场也进入同一积分。

所有有语义的五官几何均使用此方法，包括 brow=0 的情况；最初只在 brow 非零时切换的方法已因 `eyeSize=100 + browPeak 0→0.001` 的 5/255 眼部像素跳变被拒绝。证据保留在 `experiments/brow-flow-v4-zero-crossing-rejected-20261014`。

当前固定障碍只有 hair/ears/accessories，不按“某五官参数是否为非零”解冻另一五官。眉毛的眼睛保护只乘到眉毛自身速度。普通眼/鼻/嘴编辑仍会作用于各自支持域内的邻近皮肤，不宣称所有非目标五官都逐像素固定。

积分步数由局部速度、支持域、眼睛过渡和运动域栅格梯度估计，至少 32。距离域和双线性权重只是工程约束；有限 RK4、f32、光栅蒙版与像素采样没有严格的全局无折叠证明。

## 完成标准与测试用例

- 跟踪的真实检测小 fixture：`core/image-core/tests/fixtures/user-brow-geometry-20261014.json`。只保留 polygon、region、landmarks 等约 42KB JSON；纹理图由测试生成，不依赖下载、模型或 `test/results`。
- 默认测试 `detected_brow_final_mapping_stays_invertible_at_extremes_and_boundaries`：30 个当前开放几何参数的 ±100，加眼眉/鼻眉组合、硬/软选区/保护/region/固定障碍、roll30 和 tall，原82加4个对称组合，共86场景。检查实际最终 Rust mapper 的 f32 Jacobian、逆流、冻结供体、固定位置、眼域偏差、目标位移、实际像素与 alpha。
- 默认测试 `tiny_feature_family_edits_are_continuous_in_final_mapping_and_pixels`：原4组眼/眉/鼻组合加5个对称跨零组合，共9组，由0→0.001；记录全图最终映射与RGBA。要求坐标变化<0.01px、最大RGBA变化≤1/255。
- V4b新增独立圆形支持域旋转半径/面积/角度检查、实际检测eyeTilt±100中心面积/角度和polygon顶点forward面积量化、public graph非均匀alpha逐identity检查、扩展画布identity源bounds检查。前三项先在旧源码明确失败，日志保留。
- `eyebrow_reshaping_moves_brows_without_bending_adjacent_eyes`：眉峰、眉尾、粗细、长度、间距 ±100 的眼睛保护、目标像素变化和 alpha 保持。
- 可选 ignored 实拍诊断检查实际最终 mapper，包括 stationary destination、region 返回、运动域和组合场；不是 Python f64 内核推导的替代品。

命令：

```sh
cargo test --release --manifest-path core/image-core/Cargo.toml geometry::tests -- --nocapture
make native wasm
node scripts/audit_brow_flow_ui.mjs
```

## 当前进度与验收边界

当前V4b geometry模块：31 passed、0 failed、1 ignored，13.10s。受影响eyeTilt+100实际UI原尺寸native/WASM全RGBA差异0、页面错误0；5个raw非均匀alpha场景全RGBA差异0、冻结hidden RGB变化0。30个眉毛与原UI18个未受影响输出经过当前immutable worker重新渲染，48次PNG hash全与旧接受结果相同。

旧42个原尺寸mapper诊断本轮未重跑：全部零折叠、零冻结供体导入、固定运动0、眼包络偏差0；最小采样Jacobian0.0936301、逆流最大0.00921631px。这些场景均有非零眉毛且rotation=0，其速度公式未变，保留为历史证据而非新版本全量重新验证。实拍覆盖六张独立原照与一张旋转派生，派生图及不同参数不作为独立样本加计。

28 项眉峰/眉尾 ±100 和用户照片 2 项 +40，使用相同源图、相同参数和不可变旧/新 native worker 生成原尺寸裁切。眉形确实改变，眼睑保持稳定；极端眉峰仍有夸张拱形，部分眉尾或下降受眼域/遮挡约束而较弱，邻近皮肤纹理仍可被拉伸。这些几何检查通过不等于眉形、肤质或全部五官达到 8 分。

最终实拍 native 总进程耗时 408–1559ms（包含 I/O 与并发负载；初轮同批477–2520ms，不能据并发差异比较速度），不能称为实时。browser 组合导出也需数秒，后续应评估网格积分与插值/缓存，且必须复用目前的完整映射和零跨越回归。

旧19例UI是首次接受的历史结果；本轮只重跑受影响的eyeTilt UI，其他18项用fresh native hash不变核对。旧三组原照0→0.001逐像素相同的输出也在回放中核对。旧全仓Rust/E2E通过不当作本次源码的通过记录。当前源码/fixture/worker/报告及剩余眼缘面积变化见[V4b复核页](../qa/brow-flow-v4-20261014/hotfix-v4b/review.md)。

## 变更记录

- 2026-10-14：拒绝直接眼域 damping 和仅内核 SVF 候选，保留 V1/V2/V3 失败。
- 2026-10-14：加入连续运动域、眼睑包络与组合场积分，通过 42 个实际 f32 mapper 实拍场景。
- 2026-10-14：修复眉参数跨零切换，全体有语义五官统一使用 SVF；增加可在干净 checkout 重复的 82 场景和 4 组全图连续性回归。自然度与性能仍未 8 分。
- 2026-10-14：独立审计后撤回V4初次接受的旋转面积/透明字节合同，保存旧源码与报告；V4b修复旋转生成元和identity RGBA复制，默认矩阵扩大至86、连续性至9组，增加area/angle/alpha回归并重跑受影响UI与raw parity。眼缘仍非严格保面积。
