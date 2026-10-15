# FUNC-011 人脸轮廓与五官几何质量

状态：2026-10-14 v19联合脸颈已按用户授权接入正式native/WASM核心，四档原照真实UI/Go/native/WASM与旧candidate全RGBA一致，受影响10项E2E通过，见[集成对账](../qa/face-joint-v19-integration/review.md)。以下v11及隔离v14–v19记录保留为各自历史快照；脸型整体仍未达到逐功能8分验收。

用户随后明确拒绝当前瘦脸审美效果，认为不自然、曲线不好看，脸/五官/身体曲线不能统一。生产接入与字节一致只证明执行正确，不等于质量验收；本轮冻结部署保留，后续曲线候选另行对账，不恢复8分或旧数字评分。

2026-10-14 用户原照仍出现瘦脸双向轮廓裂缝，按未通过/不可用处理，旧样片与保护计数不能覆盖此失败。见 [用户失败](../qa/user-face-contour-20261014-review.md)。v12 附件连接/边缘传输候选仍有脸颈材质断接，v13 连续可逆候选虽数值无折叠却出现凹陷/鼓包，都未接生产；见 [v12](../qa/face-side-v12/review.md)、[v13](../qa/face-flow-v13/review.md)。新版方向与 Gemini 最新混合方案比较见 [路线评审](../research/hybrid-portrait-route-20261014.md)。

自行研发的轮廓/颈连接v14–v18仍为拒绝的隔离实验。v17保护/selection修复不改变用户六档失败输出；v18降低实际内收并有3项独立回归失败，颈层延拓/LaMa也未通过材质验收。完整版本、哈希和拒绝理由见[实验对账](FUNC-011-face-rbf-neck-experiments.md)。这些版本不因v19生产接入而改写成已接受。

历史v19隔离快照把可信脸和可见上颈作为同一物理前景/同一速度流传输，保留低颈静止。同一原照四档大黑楔与脸颈斜接明显改善，强档仍约20px源脸边界运动；当时原skin细岛/耳根点/硬边与双端验证未完成，见[旧v19 QA](../qa/face-joint-v19/review.md)。本轮已把同效果接入正式共享核心，质量限制仍保留。

当前v19公开operation/参数不增加experiment开关，可信skin与neck连通时默认走联合前景；neck gate须owned、faceContourMask≤.5、skin>.5且neck>.8，缺项/断开回退原有效face-only或手工路径。QP不再因chin/forehead从0跨到微小数值切换方法，竖向位移连续保留；混合模式/opacity/软选区合成同时维护semantic冻结和用户保护。最终gate源码face测试42 passed/0 failed/1 ignored，稍早全Rust206+10通过属于gate微调前快照；最终clippy/native/WASM/Go/Web及受影响10项E2E通过，旧40项隔离计数不加到生产测试数。

新默认E2E用跟踪的真实检测polygon与生成head/face/skin/neck栅格，一项测试内四档真实UI保存/导出、Go fullRGBA parity、neck变化、保护和alpha均通过，无本机忽略照片/模型依赖。可选原照±30/±100四档UI/WASM、Go API、immutable native与旧candidate全RGBA差异0；实际浏览器耳/发/配饰coverage≥.5、原ROI外、y≥700低颈及alpha变化均0，见[实际UI权限](../qa/face-joint-v19-integration/actual-ui-contracts.json)。四个强度只有一张独立原照，耳侧12px遗漏skin岛、细点与硬边仍可见，不升8分。

当前QP只在source-v排列的侧向曲线上施加单调/曲率约束，随后保留source-v并叠加解析dv插值；不能据此宣称混合chin/forehead后的最终物理外轮廓绝对无新凹凸或fold。normal保持一次几何衰减，non-normal补最终软选区/保护合成并恢复identity硬冻结。双端字节一致与有限采样数值回归不是普遍几何证明。

最新原尺寸边缘改善、三个拒绝候选和120项保护回放见 [v11 QA](../qa/face-edge-v11/review.md)。v10 物理 alpha 证据见 [v10 QA](../qa/face-physical-alpha-v10/review.md)。该轮保留头发、耳朵和配饰原输出冻结合同；发际线扩张与前景头发遮挡的排序仍未解决，不用放宽保护统计来授予8分。

真实脸缘变形、rigid MLS 与 F/B 颜色分離已实现；自然度验收仍未通过。强下巴缩短仍有颈部阴影补偿缺陷，不能以保护指标、测试通过或论文出处替代至少 8/10 的效果门槛。

最新 v3：加入有条件颜色线 alpha 校正、误冻结旧脸缘修复和连续 neck donor 补偿；瘦脸粗混色边、下巴延长点线有所改善。九参数/强值重新与 `ea5b404` 对照，详情以 [v3 QA](../qa/face-refinement-v3-review.md) 为准。该文也记录一轮 binary 被 Cargo integration 构建覆盖而作废的 120 项验证，旧混用图不得作为当前效果证据。

关联需求：瘦脸、太阳穴、颧骨、下颌、额头、下巴与五官对称。WASM 与 native 共用 Rust 核心，Go 不另写一套美颜算法。

## 用户行为

选择脸型或五官工具，检测当前调整层的输入人像，再调数值。检测不包含该层自身；上游几何或图层顺序改变后，应重新检测。自动检测失败时可手工描绘实际脸缘；仅矩形选区不具备人物前景信息，不能执行脸缘重塑。

## 轮廓输入与保留的回退基础

`engine.rs` 将 `face.adjust` 直接交给 `face.rs`，该模块自行合成一次。旧的整图局部拉伸与脸缘处归零的模板不再处理脸型参数。

检测使用 MediaPipe 定位解剖关键点，并由 BiSeNet 像素解析区分头部、面部、头发和脖子。类别概率经 `smoothstep(.4,.6)` 转为覆盖率，去除背景上的置信度尾部；概率本身不能当作物理透明度。两个有序轮廓均从对应 raster 的 alpha=.5 等值线提取，保留凹形而非凸包。

| 输入 | 用途 |
|---|---|
| `landmarks` | 有序 36 点定位解剖尺度；不能代替实际人物分割边缘 |
| `semanticMasks.faceContourMask` | 真正移动的面部类别覆盖率；供F/B颜色分离、前景反采样及旧面部边带擦除，不含头发、脖子、耳朵、眼镜和耳饰 |
| `semanticMasks.faceContour` | 上述覆盖率的 .5 轮廓；提供真正脸缘的源/目标控制线，避免仅收缩外侧发型 |
| `semanticMasks.face` | 含头发、耳朵、帽子等的头部上下文；v7起不再把这整个head alpha当作可移动脸部alpha |
| `semanticMasks.faceOutline` | 头部前景的 .5 外轮廓；提供完整背景补偿上下文，不能替代长发内侧的脸缘 |
| `semanticMasks.skin` / `neck` | 可信脸/上颈连通时定义v19联合物理前景；低颈归零，缺可靠连通时neck保留为face-only露底补偿 |
| `semanticMasks.hair` / `ears` / `accessories` | 独立冻结头发、耳朵、眼镜/耳饰的原输出像素 |
| `subjectMask` | 排除人物其他部位和衣服，防止被当成背景纹理 donor |
| `region` / `mask` | 输出编辑支持区，独立于源人物 alpha |
| `protectMask` | 用户冻结区；位移和最终合成都遵守保护 |

当前自动语义路径使用`face_flow.rs`的RBF/SVF、可行侧向QP与联合脸颈coverage传输。下面MLS/F/B和v3–v11记录属于保留的历史/face-only与手工回退基础，不能把它们描述成默认联合流的新算法。

旧核心计算有序脸缘的目标位置，沿弧长取64个逆向控制点，用Schaefer等（SIGGRAPH 2006）的rigid MLS构造反采样场。3px輪廓带精确采样，内侧MLS；4px懒加载网格、距边缘约8.7px内精确计算。眼、眉、鼻、唇作为固定anchor，五官原像素保护支持滚转和非方图，原脸缘不归零。

不透明照片先用 Germer 等（2020）的 Fast Multi-Level Foreground Estimation 分离前景/背景颜色，再按 RGBA alpha 与实际面部 alpha 预乘采样，减少边缘携带旧背景色；真实透明 PNG 保留原 RGBA 路径。算法只估计 F/B，不估计人物 alpha。少量没有直接 donor 的内部不确定区使用估计背景颜色，不能称为恢复被遮住的真实纹理。

v7将F/B与采样alpha从整个head改为面部类别alpha，避免把头发texel当作可移动脸部。固定头发可作为露出底层的已观察颜色donor；脖子、头发、场景三层单独传播，原头发输出仍保持原坐标和像素。

v8新增`refine_portrait_alpha`：先将确定前景/背景donor各腐蚀2px，再在窄边带校正已经饱和为0/1但RGB仍混色的像素；内部确定区保持原alpha。身体仍调用原`refine_alpha`入口，不共享该脸部专用策略。

v10 让 refined physical alpha 一致参与旧覆盖、holes、layer 和 donor eligibility；独立 head 上下文的确定空外部仍冻结。同一面部 tracing 才允许清理真实混合外缘。已知 F/B 回归与 72 项真实保护回放通过，但耳下 hair/scene 阶梯、黑三角和复杂隐藏纹理仍失败，不能记为自然度 8 分。

v11 仅在面部 raster、下颌下24%、距目标曲线<8px、法线内收>1px且旧曲线外6px有coverage>=.999的clean neck支持时，以目标几何alpha限制采样opacity，并同倍缩放预乘颜色；不改源matte、MLS强度和neck照明。makeup/woman的新下巴齿状皮肤凸起减少，但旧暗弧、平颈部阴影、hair黑三角仍失败。72项faceWidth/jaw/chin与额外48项chinWidth/chinPoint保护回放通过，新增两项opacity/未混色RGB真值与禁用条件回归。切线均值、精确q照明与单调法线剖面均被拒绝，未降低测试要求。详见v11 QA。

v9五官冻结按实际polygon内部+外侧2px过渡，MLS固定点也位于polygon内。旧的大enclosing ellipse及其极值固定点可能落入可编辑脸颊，在目标轮廓旁留下identity岛和凹口。新增回归检查眼旁轮廓至少真实内收6px，同时眼像素恒等；只改输出冻结、仍用旧ellipse handles的中间版本实际失败，仅移动5px，未作通过处理。

v3 在 F/B 估计之前单独校正不确定的局部 alpha：观测前景/背景的颜色线投影只有在 donor 足够近、色差足够且残差较小时采用，否则保留原覆盖率。这个模块是有条件的项目适配，不能称为 ViTMatte 或完整学习式 matting。采样用校正 alpha；旧覆盖率仍用于旧边带擦除和原背景保护，具体公式与门槛见研究记录。

纯收缩保持原面部覆盖率为零的外部像素不变，避免凹轮廓意外覆盖背景。完整原 alpha fringe 参与擦除，否则强下巴缩短会留下原下巴弧线；手工 polygon 无物理 alpha 时保守保持其原几何外像素。头发、耳朵、配饰和明确的其他非颈部人物区域冻结。补偿按最近可达 donor 分离颈部与场景背景，并仅对新露出的像素做浮点 harmonic 颜色外推，避免 byte 量化和少量迭代引起的垂直色块；两层不混色。它仍不能恢复真实颈部光照或复杂背景纹理。

v3 对 face/neck 分类过渡保留完整可变边带，避免两类低置信被误当成“另一人”而固定旧下巴暗点。扩张前景真正覆盖且距旧曲线不超过 3px 的漏分边带可被遮盖；用户/头发/耳朵/配饰保护始终优先。新露 neck 可采相应已见 clean donor，并在真正暴露宽度内连续退回重建底层，避免第二条硬弧线；原可见脖子没有被一起拉伸。强缩短照明仍未达到验收门槛。

像素轮廓须跟随相应 raster 的 .5 等值线。校验检查等值线位置和局部跨越，不要求概率在三像素内跳至 .25/.75；旧的硬边校验会错误拒绝真实软边。几何像素化允许小于一个 mask 像素的定位差。缺少匹配 alpha、轮廓不一致、轮廓过小、全画布假轮廓或缺少干净补偿 donor 时返回错误。

## 支持范围

| 参数 | 正值效果 | 负值效果 |
|---|---|---|
| `faceWidth` | 两侧脸缘内收 | 外扩 |
| `temples` | 太阳穴内收 | 外扩 |
| `cheekbones` | 颧侧内收 | 外扩 |
| `jaw` | 下颌两侧内收 | 外扩 |
| `foreheadWidth` | 额头加宽 | 收窄 |
| `chinWidth` | 下巴加宽 | 收窄 |
| `chinPoint` | 下巴侧缘收尖 | 外扩 |
| `forehead` | 额头上缘加长 | 缩短 |
| `chin` | 下巴下缘加长 | 缩短 |

`faceHeight`、`midface`、`doubleChin` 的非零值明确拒绝，尚未实现解剖重建，不能用整图拉伸冒充。手工 polygon 与旧项目关键点 polygon 可作为几何近似路径；缺少像素分割时，发丝、耳朵和皮肤交界的准确性较低。

`feature.adjust` 独立处理检测到的眼、眉、鼻、唇语义区域。对称仍是保守二维特征位置矫正，不能视为三维姿态/透视或整脸轮廓对称。五官/对称冻结头发、耳朵、配饰输出，同时检查四个非零双线性源采样像素，防止把配饰复制进邻近皮肤。妆容与肤色边界见 [FUNC-015](FUNC-015-skin-and-makeup-quality.md)。协议见 [操作 schema](../../schemas/operation.schema.json)。

## 已验证

回归涵盖最宽脸缘实际内收、九参数正负方向、额头/下巴端点、五官固定、roll/像素尺度、ROI/opacity 一次合成、用户保护、手工 polygon、第二个人不受影响、RGBA、匹配 .5 等值线、完整原面部外像素保护、全耳朵/配饰保护、下颌露脖子/脸颊露背景分层。MLS 三项独立测试核对控制点与论文矩阵公式，F/B 两项测试核对混合边恢复与透明输入限制。执行结果与真实照片当前缺陷以 [QA](../qa/face-contour-visual-review.md) 为准；性能基准是需要本地缓存的显式 ignored 测试。

真实 native 样片、当前残留和失败均记录在 [脸缘目视核对](../qa/face-contour-visual-review.md)。原图、输出和差分对比不能仅靠“脸部改变像素数量”判定通过。浏览器多样片检查另见 [多样本人脸算法记录](../qa/portrait-multisample-review.md)。

## 当前效果限制

- 分割仍会错判细发丝、耳饰、遮挡和低对比边缘。已提供独立保护层，但错误类别和漏分割仍不能保证保护准确。
- 边带重建使用二维局部 donor，脖子层选择也依赖已见像素；复杂颈部遮挡和背景纹理不能保证自然。
- 正脸与滚转样片已见真实脸缘内收，但侧脸检测有失败，细边缘仍需复核。未宣称达到专业精修软件的普适自然度。
- `chin=-100` 真实样片仍有新露出颈部偏平、阴影不吻合的补偿痕迹，效果未通过；不能把其他参数的回归结果扩展为该项至少 8 分。
- v3 九参数六图 ±60 和强下巴六图 ±100 重新回放：外扩粗混色边减弱、延长旧点线明显减少；细边、强缩短阴影/浅点仍未通过。分类覆盖率不能替代完整物理 alpha，额头加长/加宽仍受固定头发合同限制。不得用某张修复成功图替代多样本至少 8 分验收。

## 变更记录

- 2026-10-14 v11：目标下颌opacity约束只清理有neck支持的收缩碎片；120项保护通过，保留三个失败候选、纹理回归失败及原尺寸边缘对照，未授予8分。

- 2026-10-13：用独立的真实脸缘控制、头部前景 alpha 和边带补偿替换旧模板；增加像素解析、软等值线校验、长发上下文和脖子/背景分层补偿；记录真实照片效果与检测失败。
- 2026-10-13：采用 rigid MLS 与官方 F/B 估计公式、懒加载场网格、独立配饰/耳朵/头发保护，修复凹轮廓固定背景变化、原 alpha fringe 残影与补偿量化色块；五官增加源采样保护。研究来源与候选边界见 [研究记录](../research/face-warping-and-layered-matting.md)。
- 2026-10-13 v3：局部颜色线 alpha 校正、face/neck 过渡及 3px 漏分 fringe 误冻结修复、连续 clean neck donor；新增能让旧版实际失败的边缘回归。QA 改为只读按 SHA 命名的独立 executable，每次 spawn 前后及末尾校验，明确作废混 binary 的完整一轮。视觉暂评与仍未达 8 的项目见 v3 QA。
- 2026-10-14 v7–v9：实际面部alpha分层、脸部专用窄trimap、固定头发底层donor、真实五官polygon保护与固定点。新增饱和分类混合边回归及眼旁轮廓/眼部恒等回归；真实多图QA新增实际五官内部的逐像素恒等检查。仍保留强值、头发交界、额头和侧脸不足8的状态。


V4证据：[颈部照明与混合边复核](../qa/face-refinement-v4-review.md)。候选改善缩短下巴边缘浅点，但额头、太阳穴、发际线和强值自然度仍未通过。

v5使用同一neck层中的有效donor沿法线短距离桥接旧轮廓接缝；六图chin/jaw正负60/100共48项保护检查通过，新增接缝连续性、显式冻结和远端纹理回归。证据见[窄接缝复核](../qa/face-seam-v5/review.md)。发际线小带覆写候选已实际48次回放，但新露发丝重建和skin/hair接缝未通过，因此没有进入生产；源码diff和失败对照见[发际线实验](../qa/face-hairline-experiment/review.md)。
