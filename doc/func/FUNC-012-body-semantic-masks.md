# FUNC-012 身体轮廓分层重塑

状态：2026-10-13 算法整改。真实轮廓映射、背景固定、FASHN 像素分类与骨轴联合分区已接入自动检测；多图自然度尚未达到逐功能 8 分目标。最新实拍证据见 [V4 身体部位复核](../qa/body-parts-v4-review.md)，保留 [V3 分区修复与残留](../qa/body-parts-v3-review.md)，保留 [V2 失败基线](../qa/body-contour-v2-review.md)，论文和模型取舍见 [身体算法研究](../research/body-contour-algorithms.md)。

## 目标与能力边界

收腰、胯宽、手臂和腿部粗细必须改变目标的可见轮廓，人物外的原背景不参与变形。中轴、关节、保护区和非目标部位保持固定；收缩新露出的区域只重建对应窄带的底层。

当前是基于真实主体 alpha 的二维轮廓重塑，不是三维人体重建，也没有实现 SAFlow 训练模型。Pose 能预测被衣服遮住的骨骼，不能证明裙下腿部、宽袖内手臂或交叠身体有可编辑轮廓。

## 操作协议

前端滑块、工程回放、后端和 AI/MCP 共用 `body.adjust`：

| 字段 | 含义 |
|---|---|
| `region` | 归一化计算范围，要覆盖原轮廓、目标轮廓及软边 |
| `mask` | 可编辑支持域；多边形或笔刷选区，不是人物分割 |
| `subjectMask` | 非零重塑必需。真实人物轮廓 alpha，自动检测输出 raster；手工可用沿真实轮廓绘制的套索/多边形。矩形 ROI 不能冒充主体分割 |
| `partMask` | 可选目标部位 alpha，与 subjectMask 相交；能分开手臂、腿、躯干时才提供，不能以 Pose 支持域冒充部位像素解析 |
| `protectMask` | 固定手、另一肢体、饰物及其他上下文；邻近形变连续衰减 |
| `landmarks` | 最多 128 个源图归一化坐标。单肢体 2 点，腰 3 点，胯 2 点，躯干 4 点；全身沿用 Pose 33 点索引 |
| `bodyPart` | `full-body`、`torso`、`waist`、`hips`、左右上臂/前臂/大腿/小腿 |
| `adjustments` | 与 bodyPart 匹配的粗细参数，范围 -100 到 100，正值收缩、负值扩张 |
| `opacity` / `blendMode` | 仅合成实际改变像素，opacity 在身体分层内只应用一次 |

Raster 协议为 `{shape:"raster",width,height,data:[0..255]}`，坐标覆盖全图，数据长度为 width×height，最大 512×512。自动 body subject/part/protection 保持照片长宽比，最长边 512；身体分类概率当前最长边 384。

有效的非零调整缺主体蒙版、部位、关键点，或参数与部位不匹配，明确拒绝。零值不改变图像。**已删除无 subjectMask 时整片背景一起逆采样的兼容路径。**无模型时仍可手工画真实轮廓、关键点和保护区；自由推拉使用独立的手工液化，不假称语义身形。

| 部位 | 当前允许参数 |
|---|---|
| 腰 | `waist`、`belly` |
| 胯 | `hips` |
| 躯干 | `shoulder`、`waist`、`belly`、`hips` |
| 上臂 | `arms` |
| 前臂 | `arms`、`forearms` |
| 大腿 | `legs`、`thighs` |
| 小腿 | `legs`、`calves` |
| 全身 | 上述九项分别作用在各骨轴，不统一拉伸照片 |

身高、腿长、脚踝、脖长、头身比和体态等仍是后续功能目标，当前没有专用结构算法，非零调用拒绝。

## 自动检测与支持区

MediaPipe Pose Landmarker Lite 提供肩、肘、腕、髋、膝和踝等 33 点及每个人独立的分割 confidence。FASHN Human Parser（SegFormer-B4，18 类）提供背景、脸、头发、上衣、裙/裤、包、手臂、手、腿等像素概率。当前自动 body 检测先获取 Pose，再执行人体分类；不以 Pose 矩形替代实际前景，也不在模型缺失时静默回退旧的粗分割。

FASHN 不能分出同一件上衣的左右袖/躯干。`deriveBodyLayers` 在真实衣物像素内以各骨轴为竞争种子，使用图像颜色边缘加权的 geodesic 分区，生成 torso、左右上臂/前臂/大腿/小腿的 `partMask`。腰和胯使用 torso 层，目标支持域限制具体位置；裙裤腰带也可属于可见腰层。单人和多人均与对应 Pose 实例蒙版相交，减少背景椅背/挂衣被 FASHN 错归 top 后进入目标层；不使用固定骨架胶囊裁剪真实宽袖。**同类服装的 geodesic 归属是估计，不是人工真值，不能承诺重叠宽袖、垂挂外套都准确分离。**实例修正证据见 [Pose×FASHN 复核](../qa/body-instance-protection-v1-review.md)。

- 腰、胯、躯干沿骨轴法线分别测量两侧真实主体边界；支持域外扩各侧测得最大半径，避免旧猜测矩形只覆盖衣服内部而把真正轮廓固定。
- 肢体两侧独立构造支持宽度，保留非对称轮廓；真实 partMask 存在时，不再使用近侧宽度启发式把非对称宽袖剪成骨轴细条。只有缺 partMask 的贴躯干肢体才保守固定无法判定的隐藏侧。
- 胯部左右髋点定义横截面，像素空间的垂线分别转换为 normalized x/y，避免竖图的长宽比让方向倾斜。
- 裙下大腿、交叉腿、宽大裙摆遮住的小腿不开放误导的自动独立重塑；检测失败保留 unavailable。
- 脸、头发、手、包、帽、围巾、眼镜、脚、首饰由实际分类多数像素生成固定层并小范围扩边，不冻结整个 ROI。示例中的电脑被模型归为 bag，保护有效，但这不能推断所有持物都会被正确识别。前端可叠加/擦除真实 raster 保护区；模型错误仍需手工校正。

### 模型与浏览器运行

`make body-models` 下载固定 revision 的官方权重，校验 SHA，使用 `scripts/export_body_parser.py` 导出 ONNX 至 `apps/web/public/models/bodyparser.onnx`。模型属于本地忽略的资源，不把 256 MB 权重写入 Git。官方 revision 为 `1f80c34dbab321c5730dda5c3fea279fd3e97498`；ONNX SHA256 为 `3b4925715aecbf4996fea5de5e7ce30b173c20f751f21cffe826b5db46ed7729`。

- 输入：RGB，正面积缩放到 384×576，ImageNet mean/std，`float32 [1,3,576,384]`。OpenCV 与 Web 的整数 tie rounding 可差 1 个 RGB 级，见 [输入对齐实测](../qa/body-parser-input-parity.json)。
- 输出：原始 `float32 [1,18,144,96]` logits；Web 双线性重采样 logits，再 softmax 到 18 个概率 raster。V3 实拍回放使用缓存的官方 Torch 概率（先全图重采样再 softmax/缩小），两种后处理不能声称逐数值一致。实测侧身图两条端到端路径的 argmax 差 54/98304 像素（0.055%），见 [实际 Web/cache 差异](../qa/body-parser-browser-cache-difference.json)；这不是同输入 ONNX 数值误差。
- 独立 `bodyParsing.worker.ts` 承担预处理、ORT session 和推理；来源 RGBA 与输出 logits 用 transferable，模型请求串行。启用 COOP/COEP 的浏览器使用配置 4 个 WASM 线程，其余使用 1 个；不会受已初始化的一线程人脸解析 session 限制。
- [官方 Torch/ORT 同输入对齐](../qa/body-parser-parity.json)：最大 logits 误差约 1.05e-5，argmax 不一致为 0。此结果证明导出权重对齐，不证明最终分割准确率或自然度。
- [真实浏览器运行](../qa/body-parser-browser-runtime.json)：侧身 768×512 图，WASM provider、crossOriginIsolated=true、配置 4 线程，冷启动约 26.02 秒（load 7.10 秒、inference 18.25 秒）。先前一线程同图约 79.93 秒；不是控制变量性能实验，不能外推加速比。检测仍非实时，滑块编辑不重新执行人体解析。
- 模型/Worker 失败显示明确错误，保留手工目标轮廓、关键点和保护区流程；手工编辑不依赖 LLM 或人体检测模型。

## Rust / WASM 算法

1. 蒙版一次光栅化。笔刷按原顺序在有限 bbox 写入，保留羽化、擦除与反选语义。
2. 每条骨轴采样 65 个截面，两侧分别寻找 alpha=0.5 的首个真实边界，亚像素插值，再做 5 截面局部平滑。保留真实非对称；没有 partMask 的贴躯干肢体不把穿过躯干的长射线当作手臂宽度。
3. 目标半径由原半径、参数、轴向平滑区间与端点权重决定；固定半径 30% 的骨轴核心、关节和支持域边界。腰轴长取真实肩到髋长度的 65%，胯轴长由 polygon 在骨轴上的投影求得，不依赖倾斜衣物的矩形高度。
4. 骨轴被固定遮挡物穿过而缺失截面时，位移按轴向到缺失截面的距离连续锚定，避免整截面骤然归零撕裂衣物内部边；有效未遮挡截面保持相同力度。固定核心到目标轮廓采用单调分段线性逆映射，前景 RGB/alpha 一起以预乘 alpha 采样。保护区的距离场使位移平滑归零，避免冻结边突然截断十几像素而撕裂布纹。
5. 对不透明原照先使用源图颜色线约束细化 subject/target 的过渡 alpha，再使用共享多尺度 F/B 颜色估计，降低软边携带原背景颜色的拖影。前者是局部修正，不是完整 matting 网络，不能修复所有错误分区；后者估计前景/背景颜色。已知纯前景与纯背景精确锚定，透明原 RGBA 保持自己的 alpha。
6. 背景和身体底层固定。只有 `原目标 coverage > warped coverage + 1/255` 的实际新露出窄带才重建，避免把覆盖增加的混合边误判为洞。局部 diffusion 仅使用合法 clean donor；当请求窄带被固定手/袖子围住时保留请求，扩展最多 48px 的局部 crop，孤立洞使用最近 clean context 作供体，禁止固定遮挡物和剩余目标前景作供体。供体 RGBA 一起填入；没有合法供体仍明确报错。提供真正 partMask 时，手臂内侧可露出固定的躯干，外侧露出固定背景。
7. 软前景 fringe 全部随轮廓移动。旧 `alpha<0.5` 一律跳过会残留原轮廓鬼影，已改为仅固定确定背景；fringe 范围依据 raster 分辨率扩展，支持低分辨率软边。
8. 输出只合成实际发生映射的像素。中轴、支持域外及保护像素保持原 RGBA；纯收缩时原 alpha≤0.01 的已知背景严格固定，避免软 confidence 岛被重新采样进背景间隙。扩张/混合允许前景自然覆盖新背景位置，不能同时要求这些被覆盖像素不变。错误 alpha 会让实际旧衣边被当成背景而残留，仍需精确分割，不能用固定像素掩盖该问题。

窄带 diffusion 只能延续近邻颜色，不能保证缺失建筑线、格栅和复杂纹理无缝，也不等同背景去人的 exemplar 或 LaMa。

WASM 与同进程 native thread 的身体 F/B 分解使用单槽、最多 16 MiB 的缓存。只有实际来源 RGBA 和完整 target alpha 都相等才复用，不使用近似 hash；换照片、换部位或换工程前缀结果都会重新计算。身体 base mask 先编译为 `PreparedMask`，减少每像素重复解析 polygon/raster；brush顺序与反选继续保留。v16 五图15部位四档60项与旧native输出逐字节一致，四图同进程热中位从1041–1283ms降到205–265ms，首次从1086–1322ms降到337–449ms。现场并发负载、无检测/I/O，不是正式性能验收；Go若每次启动CLI，不享受native跨请求缓存。详见 [性能与字节对照](../qa/body-prepared-v16-review.md)。最终native/WASM与浏览器热调用验证仍归统一构建流程，不能由native结果推断WASM专项已通过。

## 验证与回放

代码入口：`core/image-core/src/body.rs`；支持区：`apps/web/src/core/landmarks.ts`；协议校验：`core/image-core/src/validation.rs`。独立测试包括：

- 非对称腰两侧真实移动，骨轴核心和所有原已知背景固定；水平胯及倾斜轴在像素空间方向正确。
- 真正手臂 partMask 收缩时，内侧延续躯干、外侧延续背景，所有非目标原像素不变。
- 正负收放、透明/半透明 RGBA、部位参数叠加、保护区、无供体错误、零值不变与无主体蒙版拒绝。
- 冻结附近布纹的单调连续、软 fringe 原轮廓不残留、opacity 只应用一次、混合不改变固定上下文。
- 不规则软轮廓与离散 confidence 岛的回归先重现原已知背景被覆盖，再验证纯收缩保持；22 项 `body::tests` release 已通过。新增被保护遮挡物围住的请求窄带：合法外部供体填充、遮挡物 RGBA 变化不污染填充、目标外完全保持。
- 前端三项支持区测试：宽衣腰 support 跨越真实外边、tall/wide/square 图的髋像素法线正确、非对称肢体两侧独立 support。
- 三项 `body-parts.spec.ts`：真实非对称 partMask 不被近侧启发式剪细；分类/Pose 联合分区具有非空 torso，手和 bag 不进入任何形变层，实例外/内为0/255；双色与 >0.35L 宽袖的等长 geodesic 路径不会因 Float32 舍入向上而反复灌入 heap，完整真实轮廓保留。

`BODY_RENDER_BINARY=/绝对路径/固定副本 node scripts/replay_body_parts.mjs` 使用相同五张实拍与参数 60，重新运行 Pose/部位分区，通过不可变 native 副本回放 15 项编辑。每项执行前后核对 renderer SHA，并保存来源、parser cache、graph、output hash；原 V2 输出保留。100% 裁切为原图/V2/V3/差分 x8，见 [渲染来源证据](../qa/body-parts-v3-provenance.json) 与 [逐项指标](../qa/body-parts-v3-metrics.json)。**changedPixels 和背景差分零不换算成自然度分数**，必须目视记录真轮廓改变和残留失败。`node scripts/check_body_parser.mjs` 执行真实浏览器 Worker/ORT 推理，不以缓存替代模型运行。

## 未解决问题

FASHN 分割和同类衣物 geodesic 部位归属仍会错；局部颜色线/F/B 估计不能解决错误语义。倾斜宽衣叠袖、侧身垂挂外套仍可能分给 torso，西装腰部缺失截面引起的内部硬台阶在 V4 已修；复杂窄带有细边/填补痕迹。解析模型冷推理慢，全身参数组合、高分辨率性能、所有体态都未充分验证。V3 的部位保护净改善不代表所有功能达标；实拍失败不得被合成测试或背景差分零掩盖。

## 变更记录

- 2026-10-13：删除无主体蒙版的背景 warp 路径；真实双侧截面重塑、目标部位 alpha、平滑冻结、共享 F/B 估计、完整软边移动；重做宽衣/非对称支持区并保存多图复核证据。
- 2026-10-13 V3：FASHN 18 类模型导出、校验、独立 Worker/WASM 推理与自动 part/protection 接入；修非对称真实 partMask support、新露出 coverage 判定和被固定遮挡物围住的窄带供体；不可变 renderer 回放与逐项失败评分。

- 2026-10-13 V4：修遮挡核心造成的截面位移跳变，新增连续性/效果保留回归；固定副本15项回放、逐张复核、0/15达到8分，完整保留V3失败产物。
- 2026-10-13 instance V1：单人也使用 Pose 实例 alpha 与 FASHN 相交；实拍椅背亮边 target=255→0，RGBA恢复原值且真实袖边继续收窄。五图15项合同通过，仍未自动升级自然度评分。geodesic best distance 与 heap 同用双精度，修复平坦双色宽袖的堆膨胀。
- 2026-10-14 instance-edge V14：对低分辨率 Pose instance 的低置信像素只取 3×3 邻域最大值，恢复一像素真实裤腿/衣物边缘，同时保留两像素间隙排除。5图15部位×4档共60次回放与 3 项浏览器测试通过；复杂宽袖、外套同类归属和窄带接缝仍低于8分，未合入风险较高的纹理 patch 重建。证据见 [instance-edge V14](../qa/body-instance-edge-v14-review.md) 与 [body-strength V15](../qa/body-strength-v15/metrics.json)。
- 2026-10-14 prepared V16：身体base蒙版编译、native单槽F/B缓存；60项实拍输出逐字节保持，补native缓存失效回归，效率改善不换算美学评分；浮点窄带另立实验。
