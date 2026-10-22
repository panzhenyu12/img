# 部位关系、目标控制与共享形变核心：研究核验

核验日期：2026-10-22（Asia/Shanghai）。本轮只读公开论文、官方项目和代码元数据，没有上传用户照片、下载模型或运行生成模型。范围为 2023–2026 年与单图人像／身体形状编辑有关的候选，重点核验五篇；不是系统综述，也不是全领域 SOTA 排名。

## 决策

继续验证“部位和关系描述 → 功能目标 → 共享形变 → 显露区修复”路线。近期研究支持把几何条件与原图外观分开、利用部位间关系、把几何编辑与未知内容恢复分开，但**没有一篇被本轮证据证明可以直接提供低延迟、逐参数可控、背景和原图细节可靠保持的完整美颜核心**。

本轮新增有价值的发现是 **LaTo 已有官方推理代码与公开权重**，其 arXiv 摘要仍写“接收后公开”，不能用过时摘要判断代码未发布。不过官方代码依赖大型生成网络、Qwen 与 CUDA，任务主要为表情和姿态；适合借鉴几何条件设计，不适合现在作为每帧瘦脸滑杆的强制依赖。

本项目继续采用 Rust/WASM 的确定性几何执行；AI 可提供检测、目标建议或局部修复。手工点、曲线与蒙版能够代替自动理解输入。这里的“共享”是共享映射、采样和有效性检查，**不是让眼睛、下颌和腰部套用同一个目标形状公式**。

## 检索与证据范围

先检查了已有 [2025–2026 候选核验](portrait-candidates-2025-2026-20261014.md)、[Drag/头像适用性](drag-avatar-liveportrait-applicability-20261014.md)、[单图分层研究](portrait-layer-research-20261015.md)，避免重新下载或重复跑已经失败的 Inpaint4Drag 原样几何。

本日查询记录：

| 来源 | 查询或读取范围 | 覆盖限制 |
| --- | --- | --- |
| arXiv API | `cat:cs.CV AND (ti:reshaping OR ti:"face editing" OR ti:"portrait editing" OR ti:"image deformation") AND submittedDate:[202301010000 TO 202610222359]`；按提交日期倒序，最多 60 条 | 标题筛选可能遗漏使用其他术语的工作；`reshaping` 有大量非修图噪声 |
| arXiv API | `cat:cs.CV AND (ti:"deformation" OR ti:"geometric editing" OR ti:"image warping") AND submittedDate:[202301010000 TO 202610222359]`；最多 35 条 | 多为三维重建／医学／动画，不把这些任务的性能外推至单图修图 |
| arXiv API + HTML | 按 ID 读取五篇摘要、版本、正文方法与局限 | arXiv 无期刊／会议字段不代表未被接收，需对照官方发布 |
| GitHub repository search API | `LaTo landmark face editing`、`Odo body reshaping`、`Flux-Sculptor`；补查 `Zero-shot Text-driven Physically Interpretable Face Editing` | 搜不到只能记为本次未找到，不能证明全网不存在 |
| GitHub 官方仓库 | LaTo、AAGN、Inpaint4Drag 默认分支 commit、完整 tree、README 和推理／下载脚本 | tree 未截断；未执行代码，不等于完成复现 |
| Hugging Face model API | 官方 LaTo model metadata + blobs | 只读取文件名称、大小与公开状态，没有下载权重 |
| Google HTML | 尝试三个一般检索短语 | 返回脚本挑战页，无有效搜索证据；不据此补全论文事实 |

API 入口：[arXiv](https://export.arxiv.org/api/query)、[GitHub repositories](https://api.github.com/search/repositories)、[LaTo HF metadata](https://huggingface.co/api/models/Alibaba-Research-Intelligence-Computing/LaTo?blobs=true)。以下引用都是论文或作者发布的一手来源；未核验出版的工作明确列为预印本。

## 五篇工作的适配判断

| 工作与状态 | 已核实的实际机制 | 可借鉴内容 | 不能据此承诺 |
| --- | --- | --- | --- |
| **LaTo**，2025 预印本、2026-01 v2；官方仓库标注 **ICLR 2026** | 将 68 点几何编码为 landmark tokens，与指令、参考图外观解耦输入生成器；提供 VLM 目标点预测 | 先明确“目标几何”，再决定怎样保持外观；关键点可以是条件，不应盲目当作全部精确硬约束 | 自动预测的目标必然美观、强侧脸／遮挡稳定、浏览器实时或原图像素不变 |
| **AAGN**，**WACV 2025**（官方仓库标注，预印本始于 2024） | 以身体部位为节点捕获 affinity，生成重塑 flow；判别器联合空间与高频特征 | 部位间的结构一致性；衣物高频细节要单独评价，不能只追逐位移或像素误差 | 开源目录有训练代码就等于已有可直接运行的预训练重塑模型；图关系等于真实遮挡关系 |
| **Inpaint4Drag**，**ICCV 2025** | 像素空间双向 warp 后，将显露区转换为 inpainting 输入 | 把交互形变、可见性和内容修复分开，保持可替换接口 | 官方 warp 自带不折叠、解剖约束、五官保形或固定背景保证 |
| **Odo**，**2025 预印本**，本次未核验会议接收 | SMPL 目标 depth + SDXL ReshapeNet；ReferenceNet/IP-Adapter 提供源外观 | 明确“目标形状条件”与“原图细节条件”不同，分别消融评估 | 生成结果与原照服装／背景字节相同；所用数据是同一真人真实变胖变瘦的配对照片 |
| **Flux-Sculptor**，**2025 预印本**，本次未核验会议接收 | 文本对应区域定位 PASL；按结构到细节阶段融合 latent 和 attention | 编辑权限与结构／细节阶段分开；可以作为未来生成式专项对照 | 标题中的 flow 是本产品可直接调用的二维像素位移场；mask 限定就自动保证身份与背景不变 |

### 1. LaTo：真实发布了，但不是轻量形变库

来源：[论文 v2](https://arxiv.org/html/2509.25731v2)、[官方项目](https://ali-videoai.github.io/lato_page/)、[固定代码](https://github.com/alibaba/landmark-tokenized-dit/tree/97b7331e50634e0aee8ae8ab9f25d31f12b02aca)、[固定权重目录](https://huggingface.co/Alibaba-Research-Intelligence-Computing/LaTo/tree/40c90ee4ab9d9ed9b437a33166f6f2e344136cd5)。ICLR 状态来自作者仓库标题，本轮未独立读取 OpenReview 会议记录。

官方三个阶段为 `1_get_landmarks.py`、`2_landmark_predictor.py`、`3_lato_inference.py`。第二阶段把点分成 JAW/BROWS、NOSE、EYES、MOUTH 四组，预测目标点；第三阶段直接接受 `landmark_value`，说明目标几何与图像生成确实有清楚接口。

它不是整张脸每个点都必须严格跟随的几何求解器。论文指出：错误或跨身份的几何条件，若被僵硬地逐像素执行，可能损伤身份；使用 token 条件让模型结合外观决定输出。这支持我们保留可检查、可修改的目标层，但不能证明生成器的输出遵守精确控制点合同。

论文数据主要覆盖表情、姿态与一些属性编辑；训练对混合了合成编辑和从视频提取的真实帧，不能解释成 15 万对专业瘦脸示范。论文 Appendix E 明确承认复杂模型影响即时处理，强遮挡／极端姿态降低 landmark 可靠性。

本日官方 HF API 返回 `private=false, gated=false`。主要文件大小：

| 文件 | 字节 |
| --- | ---: |
| `lato/lato.safetensors` | 24,855,692,578 |
| `Landmark_Predictor/model-00001-of-00002.safetensors` | 4,990,497,880 |
| `Landmark_Predictor/model-00002-of-00002.safetensors` | 3,885,221,448 |
| `Po_VQVAE/model.safetensors` | 177,452,832 |
| `lato/vae.safetensors` | 335,304,388 |
| `lora/lato.safetensors` | 650,495,050 |

还要下载 Qwen2.5-VL-7B；官方安装以 CUDA 12.4 / torch 2.5.1 / Linux flash-attn wheel 为例。`3_lato_inference.py` 默认 CUDA，存在循环去噪、bfloat16 和可选 offload。**不新增该依赖到当前核心，不下载这批权重。** 后续若做服务端表情／姿态生成专项，可以独立评估，不能直接继承其论文评分。

### 2. AAGN：关系意识有价值，开源推理链仍不完整

来源：[论文 v3](https://arxiv.org/abs/2404.13983v3)、[固定仓库](https://github.com/Randle-Github/AGGN/tree/312176f9cbada0b0b263c32fa421539c721eb536)。论文名称中的网络为 AAGN，仓库名为 AGGN。

本日默认分支 SHA 与此前核验相同，完整 tree 未截断。README 指向 `evaluate.py`，根目录仍未见该文件；提供 RTMDet/RTMPose 权重链接，不等于提供身形重塑网络 checkpoint。保留“方法参考，不直接新增模型依赖”的决定。

自适应图学习的是特征相关性；本产品的 `connected / occludes / follows / fixed` 关系不是论文直接输出的结果。我们可以研究关系分组能否减少腰／臂交界冲突，但应明确为本项目设计，不称为 AAGN 复现。

### 3. Inpaint4Drag：采用职责拆分，保留既有负例

来源：[论文](https://arxiv.org/abs/2509.04582)、[固定官方代码](https://github.com/Visual-AI/Inpaint4Drag/tree/bfb0fafd456ac01ee23ade0cc04b64c9781eaa87)。本日默认分支与此前一致。

已有本地 [原样组合负例](../qa/inpaint4drag-20261014/report.md)：目标 mask 相近仍会错误带动鼻口，补全可能把收进去的脸重新补宽。无需为引用“前沿”重复跑原样方案。

继续采用 warp／显露区／修复接口。模型只在真实未知、允许写入的目标区域提供候选；露出颈部与露出场景需要不同上下文。论文作者在 512² 的阶段耗时不能当我们的全分辨率／浏览器数字。

### 4. Odo：目标语义相关，完整生成链不适合目前交互核心

来源：[论文 v3](https://arxiv.org/html/2508.13065v3)、[arXiv 记录](https://arxiv.org/abs/2508.13065)。本次论文链接和 bounded GitHub 查询未找到可核验的作者推理仓库／专用权重，状态是“未找到”，不是断言永远未公开。

论文 §4–5 使用目标 SMPL depth 作为身体形状条件，原图由冻结 ReferenceNet 和 IP-Adapter 提供外观信息，ReshapeNet 负责生成。它采用 SMPL 身体先验，与用户已排除的 FLAME 头部模型不是同一个模型；但要真正采用仍需新增身体拟合和模型链，本轮不扩展。

数据 §3 从 FLUX/PuLID 生成人物和 thin/fat/muscular 变化，再做背景归一化。尽管包含来自图库的人脸参考，**身体编辑配对主要来自生成管线，不是真实人物身形改变的摄影真值**。测试集也沿用同类构建方式，对我们真实照片、复杂衣纹的泛化需另验。

论文自报推理约 23GB 显存、18秒／图，训练使用 A100 80GB；这不是我们机器实测。作者也承认衣物／文本细节重现的局限。可学习其“形状条件与外观条件分工”，不采用为客户端实时底座。

### 5. Flux-Sculptor：局部语义生成对照，暂不依赖

来源：[论文](https://arxiv.org/html/2507.03979v1)、[项目页](https://flux-sculptor.github.io/)、[项目页源码](https://github.com/Flux-Sculptor/Flux-Sculptor.github.io)。官方示例包含 slim lower face / chubby face，任务相关性比单纯 talking-head 动画更直接。

但本次作者页面与 GitHub 查询只核实到展示网站，未核实完整推理代码及专用权重。其“Spatial Flow Control”是围绕生成模型的区域／阶段控制，不能当作我们需要的可逆像素流或现成网格引擎。

结构到细节的区分有启发：先确认目标几何和编辑范围，再评价细节修复。其图示成功样片不证明对本项目用户图或连续滑杆单调响应有效。

## 本轮只推进两项可证伪实验

下述为本项目研究性设计，借鉴论文机制，不冒充上述工作的完整复现。

### 实验 A：部位可动保形与连续连接，先做整体脸宽

**假设：** 当前不自然和部分不可行约束来自“保持部位形状”被误写成“固定原图位置”。将目标形状和部位运动规则分开，可在合理缩窄时保留眼眉／鼻口形状，并减少求解冲突。LaTo 的几何／外观拆分和 AAGN 的部位关系提供设计依据，不提供效果保证。

- 目标控制器输出明确的轮廓目标与部位目标变换，不从已修改图继续累计变形。
- 分别记录 `target_geometry`、`shape_preservation`、`position_preservation`、`texture_source`。允许跟随的眼眉保留形状并移动；背景才要求保持位置。没有额外显式五官操作时，限制眼形尺寸变化。
- 一个连续面部表面协调求解。脸／颈连接和耳根用连接关系；眼眉不作为各自独立贴纸拼回。真正存在相对运动和遮挡时，才创建独立可见层。
- 使用少量控制点／曲线和部位变换驱动共享连续场；先验证目标形状，再决定网格、样条或速度场实现。避免一开始把大量图像像素都增加成求解自由度／精确等式。
- 在同一张图、同一个目标、同一个 alpha 和修复设置下，对照原固定位置规则与可动保形规则；另以相同控制器比较执行器，防止同时改目标和求解器却说不清哪项有效。

观测：真实轮廓位移、眼眉／鼻口自身形状变化、耳颈连接、背景、整体比例，以及目标生成／求解／采样各阶段耗时。完整图片视觉检查仍是主判据；全域几何审计用来拒绝折叠，不能替代自然度。

失败／淘汰：正常档依旧有可见接缝、五官明显变形，或改善只能靠接近零效果；未参与调参照片明显退步；新增复杂度只降低数学能量而无可辨别视觉收益。明确失败后改目标／关系模型，不无限增加迭代。

### 实验 B：遮挡关系与显露域单独审计

**假设：** 部分贴片感来自源层归属和补全范围错误，而非形变平滑度。Inpaint4Drag 支持将两类问题拆开检验。

- 在几何确定之后，显示原可见材质移动到哪里、何处新显露、何处由前层覆盖。区分“连接缝”与“遮挡边”，不要一律做羽化。
- 先冻结修复模型／输出检查几何；再冻结几何比较修复。这项实验先复用已有脸／颈和手／衣袖挡腰样本，不新增大模型。
- 对未知区域允许模型候选；对已观察、无需变化的区域保持原图。实际显露区可有上下文扩张用于模型输入，但写回权限应由最终语义和可见性决定。
- 单独报告几何结果和最终合成结果。若几何正确而补全错误，就记录为补全失败，不能重调形变把缺陷藏起来。

失败／淘汰：补全重建出旧脸／旧身体，前景衣纹粘到背景，连接处依靠宽模糊带掩盖断裂，或者错误冻结遮挡物导致其与主体分离。不能用大面积低对比掩盖视为改善。

## 本轮不做的推论

连续网格只保证共享边拓扑时的位置一致，线性三角形插值一般只是 C⁰；不能直接声称 C¹ 或“裂缝归零”。平滑惩罚也不自动保证无折叠。保直软约束不能保证任意强度下所有门框绝对不弯，互相冲突的硬条件还可能导致不可行。

原图 warp 能保留已观察纹理，但缺失背景和被遮挡材质仍有不确定性。生成模型、较新论文、公开权重、单个指标领先都不能代替本产品多图双向、多档和实际耗时验收。**没有新增 8 分结论，也没有更新部署。**
