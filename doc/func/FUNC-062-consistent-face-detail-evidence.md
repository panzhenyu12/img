# FUNC-062 同一次解析的脸宽控制域与源材料证据

日期：2026-11-13。状态：隔离算法候选，16 项源证据/局部应用定向测试通过；
三图 v2 对照及 dark v3 回放已完成。局部源缺陷有改善，不能据此宣布瘦脸可用或达到 8 分。

## 已确认的问题

现有解析在较大裁剪生成 `faceContourMask/faceContour`，再从更细裁剪补充
`ears/accessories`。原脸部控制域没有随细类一起更新，可能仍包含细类已经识别为耳朵
的区域。原 `hair/skin` 也可能与细类不一致。

dark 原控制线伸入耳部；旧 hair 把一部分源耳内皮肤判为头发。只加耳朵蒙版，
或只更换几何控制线，均不能证明材料阻挡、光学估计及最终成片已经一致。

## 本轮实现

- `experiments/face-attachment-motion-20261110/consistent_detail_control.py`：
  验证完整、有限、归一化的 19 类后验；在覆盖原 landmark bounds 且脸/耳/附件均未
  被内部裁剪边截断的推理 pass 中选分辨率最高的一个。三个域均由这一个后验、
  同一个源坐标变换得到，不独立挑选不同版本。
- 脸宽控制类仍为原集合 `[1,2,3,4,5,10,11,12,13]`；耳朵为 `[7,8]`；
  附件为 `[6,9]`。不把 FaceOval 当成真实轮廓，也不利用变形输出拟合控制域。
- `prepare_consistent_detail_control.py`：复用精确源图、原图 graph、模型身份和
  裁剪几何一致的完整缓存；没有该缓存时，仅运行已有模型的源图推理。
  轮廓提取直接执行现有 TypeScript `alphaOutline`，未发明新的轮廓平滑参数。
- `driver-revision.json` 保存独立控制域及同版耳/附件证据。
  `driver-only-graph.json` 仅供生成几何目标，实际仅更改
  `faceContourMask.data` 和 `faceContour.points`。原 dark/woman graph 缺少的
  `source` 尺寸另按源 PNG 补充。**这个 graph 不得进入物理源层准备或渲染**。
- `raw-source-role-sidecar.json` 单独保存 hair `[17]`、skin `[1,10,14]`、
  neck `[14]`、ears `[7,8]`、accessories `[6,9]` 的同版提案；本生成器没有把它们
  应用到 runtime graph，也没有修改源 RGB、head、subject 或物理 alpha。

## 裁剪和低证据的边界

精细裁剪足以包含脸与耳朵，不等于包含全部头发或脖子。三张图的 hair 在精细
裁剪边界均被截断，user 的 skin/neck 也被截断，因此没有任何一张提案获准全图
替换这些材料。

源角色有效条件为：源采样位置在所选裁剪内，且**逐类投影后验后再取最大值**
不小于 0.6。无效位置明确写为 `null`，而非以 0 冒充“没有该材料”。
0.6 沿用解析 coverage 的确定区边界，不声称这是校准后的正确率。
后续若局部应用提案，应保留域外/未知的旧值，并记录这是一张混合来源图，不能
称为全图同版或静默补零。

测试先复现了 `bilinear(max(class))` 的错误：两个相邻高置信但不同类别的中点
会被误判为高置信。现使用 `max(bilinear(each class))`，跨类边界低证据继续未知。
v1 的几何控制域没有因此改变。本节记录 v2 的置信门控实验；后续 v3 已明确
区分“观测范围”与“分类置信度”，见文末，不以 v2 门控作为最终方案。

## 输入与源证据

模型为已有 `apps/web/public/models/faceparser.onnx`，SHA-256：
`1d5c459e3d4691eb6120994fd36a8d360fe3732997d1bbf245278145cdd11d20`。
使用原自动检测 graph，未接入后来人工 polygon、附件修正或 MEMatte 修正缓存：

| 样本 | 源图 | 原始 graph | 后验来源 |
| --- | --- | --- | --- |
| dark | `test/results/native-face-contour/portrait-dark-source.png` | 同目录 `portrait-dark-graph.json` | 精确复用 `face-usable-20261112/dark-parser` 完整 19 类缓存 |
| user | `test/results/user-face-contour-20261014/768/canonical-source.png` | 同目录 `face-graph.json` | 当前模型源图原生推理；旧 1024 缓存不冒充 768 结果 |
| woman | `test/results/native-face-contour/portrait-woman-source.png` | 同目录 `portrait-woman-graph.json` | 当前模型源图原生推理 |

user 的源图/graph 哈希与原 `provenance.json` 检测导出记录精确匹配；dark 的
原 graph 哈希与完整 parser 缓存报告匹配。woman 使用历史原始 graph，未找到同等
完整的原始检测导出 manifest；本轮新推理与输入身份均记录了哈希，此限制保留。
`manualEvidence=false` 指本次自动推理及原图输入没有使用人工修正，不是对语义
正确率的认证。原生 Pillow/ORT 采样不宣称与旧浏览器 Canvas 字节一致。

输出固定在忽略目录 `test/results/face-usable-20261112/consistent-detail-controls-v2/`。
每例记录 source、graph、模型、parser、准备脚本、后验版本和产物哈希。
v2 从已有完整后验重出，不重复运行模型。

| 已验证项目 | dark | user | woman |
| --- | ---: | ---: | ---: |
| 所选 pass | crop-1 | crop-1 | crop-1 |
| 控制域改变节点 | 1318 | 2936 | 884 |
| 轮廓顶点 | 130 | 116 | 118 |
| 半覆盖控制域连通分量 | 1 | 1 | 1 |
| 控制域与耳/附件同时大于 128 的节点 | 0 | 0 | 0 |
| 源角色有效节点 | 67741 / 175104 | 118700 / 262144 | 68018 / 175104 |

独立字段比较确认：原源图、原 graph 均未修改；driver graph 除声明的两个
控制字段及必要尺寸元数据外，其余材料/光学字段逐值不变。三张源图的橙旧/青新
轮廓叠图均已目视检查：dark/woman 控制线排除了旧线伸入的耳部；user 的额头
轮廓更内收。这里的“更内收”是源语义变化，不能直接解释为美观收益。

woman 附件后验峰值仅约 0.309，有效附件存在节点为 0。这不证明没有附件；
本轮不为它补造附件，也不提升自动附件检测评价。

## dark 耳内事后检查

`audit_consistent_detail_source.py` 仅把上一轮人工耳内声明当作事后评价区域，
不把该 polygon、45 个旧修正节点或变形结果传给新推理、控制域或材料生成。

- 上轮人工清 hair 的 45 个节点，本轮自动提案均为 hair=0、ear=255，45 个均有效。
- 声明区域共 645 个源像素，645 个均有完整有效采样支持。
- 旧 hair≥0.5 为 389 个源像素；新 hair≥0.5 为 0；新 ear≥0.5 为 645。
- 源 RGB/旧 hair/新 hair/新 ear/有效域的叠图已目视，与耳内皮肤相符。

这支持“同版精细后验自动修正了这一个源耳内错分”；不能扩展成所有耳朵、
附件或变形边缘已修复。证据在 dark 子目录的 `posthoc-ear-source-audit/`。

## 定向验证与后续验收

v2 阶段 `test_consistent_detail_control.py` 的 9 项测试通过：耳朵与脸控制域同步、附件同步、
任何控制类别裁剪时拒绝混版、非法后验/无共同 pass 失败、等概率不产生双不透明、
坐标投影、裁剪外 hair 明确未知、低置信明确未知、跨类别插值明确未知。
测试覆盖 source 输入不被原地修改。未运行整套应用 build。

下一步是保持求解器、采样器和目标强度固定，分开比较旧/新 driver 与旧/新 source
role。局部材料提案必须从自动原图 graph 生成新 trace/cache；不可沿用人工修正的
旧 alpha/blocked 结果，再把成片差异归功于自动算法。最终仍须检查真实脸型曲线、
耳颊连接、发际线、附件与背景，不以源分类数字代替成片验收。

## 自动回放后的剩余问题：耳边亮缝已定位，未解决

对主任务生成的 `consistent-source-replay-v2/dark` 现有输出做只读诊断。
这里 B 为同版材料配原 driver，BD 为同版材料配新 driver，BD-W10 另采用固定
耳共旋约束。未运行新模型、求解或构建，未改 alpha、背景层或算法。

在 B 的 +30 结果中，右耳可见浅色点的目标像素索引为 **(488,483)**，逆场采样
源图 **(490.0127,483.0145)**。源 RGB 插值为 `[140.19,83.45,74.51]`，属于耳缘
皮肤；实际可动 alpha 在此为 0，因此输出恰为目标位置的配对背景
`[191,185,181]`，相比同一运动场直接取源 RGB 平均增亮 **86.28** 色阶。
实际残差合成公式复现该像素，取整后误差 0。这不是网格漏画出的空白像素。

源 (490,483) 的原同版后验仍为 ear≈0.809、最大单类≈0.743、hair≈0.0145。
但 runtime ear=0、color alpha=0、scene alpha=0；该处以及目标点的 frozen/blocked
均未阻挡采样。进一步追踪到语义节点 (245,241)，其源中心约
(489.7822,482.5)：sidecar 有效且 ear=255，实际 runtime ear 却为 0。
局部应用器的 3×3 有效域收缩，因为相邻未知节点而放弃该已知节点；原 graph
缺少 ears，保留旧行为就成了零支持。相邻 (245,240)、(245,242)、(244,243)
也存在同类丢失。**记录“未知保留旧值”仍不足以保护耳缘：旧缺类的零支持进入
下游后仍可能被当成不含可动耳材料。**

| 臂 | 定位目标像素 | 源 alpha | 相对同场直接 RGB 的平均增亮 | 源点 Jacobian 奇异值 |
| --- | --- | ---: | ---: | --- |
| B +30 | (488,483) | 0 | 86.28 | 1.1600 / 0.9750 |
| BD +30 | (489,483) | 0.16354 | 53.84 | 1.1064 / 0.9972 |
| BD-W10 +30 | (488,484) | 0.03607 | 75.42 | 1.0056 / 0.9787 |

三个定位点的 Jacobian 行列式均为正；W10 在该处接近刚体仍出现亮点。因此这处
亮缝有明确的源 alpha/配对背景解释，不能只靠继续加大耳部保形约束解决。
同一逆场直接采样 RGB 的诊断图未出现该亮点，但它没有保护背景的完整合成能力，
不是新的合格修图方案，也不作为输出验收。

缓存 alpha 与本次 native color alpha 逐值相同，背景与本次 native paired
background 逐像素相同，排除了挂错旧缓存这一解释。native color/scene alpha
全局并非完全一致；本次逐个记录了定位源采样四邻点，在 B 的这些点上两者均为 0，
没有用全局等同假设代替采样追踪。

定位图和逐点数据保存在 `source-white-seam-audit/`：
`located-seam-source-alpha-background.png` 标出源点/目标点并列出 alpha/B；
`same-field-composition-diagnostic.png` 对照相同几何的合成差异；`report.json`
记录四邻权重、RGB、两种 alpha、语义、posterior、frozen/blocked、Jacobian
与文件哈希。两张图均已目视核对。

当前需要修正的是“有效/未知源类别如何进入 runtime 材料及 matting”的契约，
并重新验证耳缘光学覆盖。单个像素诊断尚未解释全部耳饰与发际线瑕疵，本轮
不得据此宣布整体瘦脸已经可用。

## v3：区分已观测的不确定类别与未观测区域

根据上面的源采样证据，新增显式 `validity_policy='observed-support'`，CLI 使用
`--validity-policy observed-support`。默认 `confident` 策略仍可复现 v2 对照。
变化只发生在源角色提案的有效性契约：

- 完整 19 类后验已经通过有限值、范围、归一化校验且坐标处于裁剪内，就属于
  “已观测”。类别分不清仍保留各角色原有 coverage 软值，不再因最大单类低于
  0.6 而变为 `null`。这没有把耳朵强制设为 1，也没有修改原 coverage 转换。
- `null` 仅表示裁剪外未观测；原应用器的裁剪边界/相邻插值支持规则保留。
  本修改没有触碰 `apply_consistent_source_roles.py`。
- 每个角色的投影后验和最大单类后验另存为诊断，标明不是校准准确率，也不是
  物理 alpha。`physicalAlpha=false`、`manualEvidence=false` 保持不变。
- 规则对整个已有裁剪统一计算；不读取白缝坐标、旧人工 polygon、节点列表或
  变形输出选择更新区域。

首轮生成 dark 的 v3：
`test/results/face-usable-20261112/consistent-detail-controls-v3/dark/`。
完整复用旧 parser 缓存，**零新推理**。`v2-v3-input-parity.json` 验证：

- driver-only graph 与 v2 **字节相同**，SHA-256 为
  `fc04899d007394496cb092f4d99d2b80389ab7aae7a9cc1c95d2245a0e9d22bc`；
  全部 driver 蒙版和轮廓逐值相同。
- 原 source、原 graph、所选 posterior 三个哈希均与 v2 相同。
- v2 原有有效节点的所有 role 字节保持不变；仅新增 2219 个已观测但原先
  被置信门控排除的节点。105144 个裁剪外节点继续为未知。

新增两项行为回归先失败后通过：皮肤/耳朵各 0.5 的交界在 observed-support
下保留约 127/128 的软值和 0.5 的后验诊断；裁剪外仍为 `null`。连同原有
9 项测试，源提案当前共 **11 项通过**；局部应用器另有 5 项来源/插值支持回归。

## v3 实际原生回放

主任务已在相同原图、driver、求解器、采样器和强度下生成新的 B/BD 0/±30，
以及固定 W10 的 BD ±30。新 cache 与 fresh native 的 alpha/背景逐值匹配；
B/BD 零值与源图完全相同。既有 A/D 输入经身份比较后复用，没有重新推理。

源观察点 (490,483) 的 alpha 从 v2 的 0 恢复为约 .9002，右耳边已不再按纯
背景处理；左侧 (221.292,479.065) 从约 .0584 恢复为 .9131。
双向整脸对照中，原耳边亮色裂纹减轻。左耳饰邻域 (227.228,513.711) 的 alpha
仍只有约 .0525，不能据几个点推断所有孔隙/混合边已经正确。

完整对照在 `consistent-source-replay-v3/dark/v2-v3-comparison.png`。
随后 user/woman v3 四臂回归也完成，旧 A/D 经身份验证后复用；新 B/BD
零值等源 RGBA，7 个五官核心不变，cache 与 fresh native 光学值匹配。
实际双向放大检查未见新增明显白缝或碎裂。

但 woman 出现执行能力回退：在相同新 driver 下 BD RMS 由约 .002182px
变为 1.452074px；原 driver 下 B 响应由 .999638 变为 .748644。输入轮廓请求、
网格、平滑及正则均不变，新源的固定节点 6940→6980（新增 85、释放 45），
非轮廓固定锚 230→243。新源类别会同时影响几何权限，不能把这轮变更说成
只换了光学层。user 的对应指标略改善，也不能抵消 woman 的回退。

因此 v3 保留为源契约研究候选，**不作整体通过或默认替换**。需要解决材料
证据与可动范围冲突后继续验收；本轮没有更改正式 Web/WASM 路径。

独立复核的 woman +30 控制点 33 `(202.907,463.803)`、93
`(434.416,428.982)` 均请求约 4.505px，v2 基本实现，v3 实际位移为 0；
相关双线性单元从 0/4 固定角点变成 4/4。需要分离头发身份与运动许可，
不能退回丢弃不确定类别证据，也不能仅放宽求解器来掩盖新的硬固定冲突。

随后按完全相同的 observed-support 策略生成 user/woman 的 v3 源提案，目录为
`consistent-detail-controls-v3/{user,woman}/`，未改算法、参数或 driver，未增加
照片或推理。每例 `v2-v3-input-parity.json` 均核实 source、原 graph、完整后验
身份与 v2 一致，driver-only graph 字节一致，旧有效角色值不变。
user 新增 5909 个已观测节点，woman 新增 3271 个；两者裁剪外继续为未知。
woman 的附件后验峰值仍约 0.309，没有补造附件。此段记录源输入迁移完成，
不代替两张图的实际成片回放与独立视觉验收。
