# 像素蛋糕公开手册功能审计与本项目算法映射

日期：2026-10-15。范围是功能证据和实现缺口，不是竞品效果实测，也不是本项目的完成声明。**当前优先级仍是让脸型、五官、身形与修复真正可用；不以新增菜单或 MCP 接口代替算法。**

## 阅读范围与证据

- 官网 [使用手册](https://www.pixcakeai.com/guide) 桌面目录共 **92 个唯一 URL，全部抓取成功，0 个失败/空正文**；去除父目录返回首子文章等重复内容，共 **83 份不同正文，已全部阅读**。
- 已核对所有桌面图像编辑章节，包括皮肤、妆容、头发、全身、手脚、衣物、背景、修复、调色、创意工具与工具箱。图示/视频只记录 URL，**未逐一查看媒体**；正文未写出的细节不推断。
- 另记录全站 **401 个目录节点**。iPad 43 个、手机 67 个（均含根节点）只读取目录，没有读取其正文；云工作流、联机、ERP 等其他栏目亦只列目录。这份报告不称“全站所有文章已读”。
- [结构化证据](sources/pixelcake-manual-functional-audit-20261015.json) 包含目录、83 份正文的对应 URL/时间/SHA256/标题与段落行号、映射记录、代码快照哈希。完整正文留在 `.run/pixelcake-manual-audit-20261015/pages/<id>/body.txt`，报告只保留功能摘要。
- 父节点与实际文章标题不总一致，例如“桌面版使用手册/快速了解”实际返回工作区文章。附录将同文引用标为“同文已读”，不重复计算成不同文章。

## 这次对账的结论

现有代码有共享 Rust WASM/native 执行、操作图与部分有效算法，但距离“功能相近”仍有明显差距：正式目录只有 10 项脸型、37 项五官/局部颜色、5 项皮肤、9 项身体、5 项衣物注册滑杆。像素蛋糕把同一部位拆为几何、颜色、材料、光照、修复与生成等不同问题，不能共用一个变形或模糊函数冒充。

尤其要分清：瘦脸与脸宽、小脸、短脸并不相同；身体胯宽不等于立体美臀；唇色不是13种口红材质；眼睑几何不等于双眼皮生成；细纹磨平不等于去粗褶；自动分割不等于物理抠图；手工圈选 LaMa 不等于自动识别并去除路人。当前已注册参数也尚未全部达到自然度 8 分。

以下“我们的策略”均是本项目的候选实现方向，**不是对像素蛋糕内部算法的断言**。手册里的“AI、3D、中性灰、无损”等是产品公开描述，没有提供训练权重、算法细节或可重复测评；不能凭词语认定它采用 FLAME、SVF、扩散、网格或某个论文模型。

## 当前代码证据索引

| 代码/文档 | 可确认的能力 | 不应据此声称 |
|---|---|---|
| `core/image-core/src/face.rs`、`face_flow.rs`；FUNC-011 | 人脸目标轮廓与连续场、分层/脸颈处理、正负方向合同 | 整体脸型自然、耳饰附着与所有遮挡均通过；父代理的新整体模式仍为实验 |
| `core/image-core/src/geometry.rs`；FUNC-020 | 五官语义定位、连续速度积分、眉眼保护；手工液化/修复 | 全部五官审美已达标、所有非目标区域都严格固定 |
| `core/image-core/src/body.rs`；FUNC-012 | 骨轴+真实主体/部位mask的粗细调整、背景分离 | 身高腿长/天鹅颈/直角肩等已实现；身体三维重建 |
| `core/image-core/src/color.rs`；FUNC-013/015 | 皮肤引导基底与纹理、肤色、局部唇/眉/牙；衣物细纹 | 独立自动祛痘、结构眼袋、自动妆容、粗褶/衣物版型 |
| `inpaint.rs`、`model_inpaint.rs`；FUNC-014/016/018 | 手工heal/clone、传统补全与客户端/服务端LaMa | 复杂建筑自然度、自动路人选择、提示词生成已完成 |
| `color.rs`、`film.rs`；FUNC-005 | 8-bit sRGB调色、曲线/HSL、14款风格与CUBE | RAW/ICC/16-bit流程、富士官方算法标定 |
| `validation.rs:615`附近 | 未验证参数非零会拒绝，避免无效滑杆伪完成 | 参数曾出现在旧工程就意味着可用 |
| `services/api/internal/api/tools.go` | `inspect_asset/propose_operations/apply_operations/render_preview/export_image/undo_operations`；共享目录 | 服务端检测已可用：当前`detect_landmarks/detect_regions`仍标记不可用、运行时为浏览器 |

`doc/feature-list.md` 是产品需求目录，不是完成清单；`FUNC-010` 是注册参数台账，也不代表8分。最新算法实验与本次静态审计并行推进，最终上线状态需以对应 func、实际构建及 QA 为准。

## 逐项功能 → 算法映射

同一行列出共用底层且需分别验收的子功能；“部分”只表示已有相关代码，不能算功能完全覆盖。`atomicIntent` 是将来提供精细 MCP 的语义名称，不是此次新注册的工具。具体来源页面与每行 ID 均同时收录在 JSON 中。


### 人物与范围

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-001 女、男、儿童、长辈女、长辈男；单人；所有人；部分属性联动<br>[47211](https://www.pixcakeai.com/guide/47211) [47248](https://www.pixcakeai.com/guide/47248) | 按人物属性或实例选择；单人覆写后原属性组不再影响，恢复属性会清除单人覆写 | 部分：Web 可检测/选择人脸与身体实例；无完整年龄组、跨照片同人继承 | 实例 ID 与人工可改属性分离，编辑只绑定实例/显式集合，避免把属性判断当身份真值 | 多人交叉、不同年龄组只修改指定实例；覆盖/恢复结果可重放 | `portrait.target.resolve` |
| PCA-002 人脸增删；年龄性别纠正；同人跨图同步<br>[47211](https://www.pixcakeai.com/guide/47211) | 允许人工修正检测并选择批量同步 | 缺口：无完整同人匹配和手册式属性继承 | 检测结果版本化，人工 landmarks/蒙版为有效输入；跨图重识别需独立证据及可改匹配 | 错误检测可纠正；未匹配人物不得套用；几何变化后坐标不陈旧 | `portrait.target.correct` |

### 脸型

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-003 脸型预设：经典、幼态、英气、少年、圆脸、长脸、方脸<br>[32466](https://www.pixcakeai.com/guide/32466) | 组合脸型与眉眼鼻嘴参数，统一强度；各风格目标不同 | 开发中：FUNC-024 整体脸型；当前正式 face.adjust 没有七风格等价实现 | 依据原脸解剖比例构造整体目标曲线，再联合头/耳/颈/发附着求解；预设必须是有语义的组合 | 同强度跨脸型不能得到同一模板脸；整体与局部不重复累加；身份和附着连续 | `face.shape.preset` |
| PCA-004 瘦脸；脸宽<br>[32466](https://www.pixcakeai.com/guide/32466) | 瘦脸作用太阳穴以下；脸宽双向；都支持左右解锁 | 部分：face.adjust.faceWidth 合并了两者；face.rs，FUNC-011 | 拆成轮廓收束与横向比例两种目标，左右独立约束；共享物理界面、单次源图采样 | 正负方向、可见轮廓位移、曲率、耳饰附着、远背景逐像素；不能只测像素变化 | `face.contour.slim / face.proportion.width` |
| PCA-005 太阳穴；颧骨；下颌<br>[32466](https://www.pixcakeai.com/guide/32466) | 均可左右解锁；太阳穴/下颌向右外扩，颧骨向右收窄 | 部分：temples、cheekbones、jaw 已执行但自然度未验收 | 分别使用颞部凹曲线、颧弓宽峰、下颌转角约束；将 UI 方向显式映射到语义 | 三个部位独立有可测几何响应且不互相冒充；遮发边界、侧脸对照 | `face.contour.temple / cheekbone / jaw` |
| PCA-006 AI面部流畅；V脸；小脸；短脸<br>[32466](https://www.pixcakeai.com/guide/32466) | 流畅处理凸凹；V脸可左右；小脸整体缩放；短脸改变耳下长宽 | 缺口：不能用 faceWidth 代称四个功能 | 曲率约束清除局部凸凹；V脸改变下半脸轮廓；小脸联动五官；短脸按下半脸长度重排 | 轮廓无折角；眼口比例保持合理；小脸与窄脸效果不同；负向可逆趋势 | `face.contour.smooth / face.shape.v / face.scale / face.length.lower` |
| PCA-007 额头宽度；额头高度；发际线<br>[32466](https://www.pixcakeai.com/guide/32466) | 额头宽高支持左右，发际线上下移动 | 部分：foreheadWidth/forehead；发际线独立材质重建未有 | 额头皮肤与发际线共享边界；高度是皮肤跨度，发际线另含毛发显露和补全 | 皮肤/头发无重复边或断层；额头调整不得缩放眼睛 | `face.forehead.adjust / hair.hairline.move` |
| PCA-008 人中；中庭；下庭；下巴长短；尖下巴；收下巴<br>[32466](https://www.pixcakeai.com/guide/32466) | 长短双向；尖下巴支持左右；收下巴同时覆盖下颌与下巴 | 部分：philtrum、chin、chinPoint；midface/faceHeight 非零拒绝；下庭和收下巴无独立实现 | 口鼻/上下颌局部坐标系约束比例；下巴与颈部阴影作为同一附着问题 | 长度控制确实移动指定端点；牙齿嘴唇不撕裂；颈部旧轮廓无残影 | `face.proportion.adjust / face.chin.adjust` |
| PCA-009 颌面矫正：轮廓塑形、光影过渡<br>[32466](https://www.pixcakeai.com/guide/32466) | 几何与光影各有子强度，主滑杆联动 | 缺口：noseStraight 不是颌面矫正 | 独立颌面结构估计与局部反射/照明修正；不可靠侧面及遮挡先拒绝 | 嘴唇突出度有改变；牙缝、唇纹、鼻唇接缝无假影；几何/光影可分测 | `face.maxillofacial.correct` |
| PCA-010 3D塑颜；五官对称；上半身对称；面部提拉<br>[32466](https://www.pixcakeai.com/guide/32466) | 3D支持上下/左右/倾斜；提拉分全脸两档及轮廓/眼周/面中/口周 | 部分：二维 symmetry；3D、上半身、提拉未实现 | 二维对称明确限制正脸；提拉用区域附着与肌肉走向；真正转头需单独可见性/重建模型 | 正脸/侧脸分层验收；不可把水平压缩叫3D；闭塞暴露必须处理 | `face.symmetry / portrait.pose.adjust / face.lift` |

### 眉毛

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-011 眉头长短；眉尾长短；粗细；眉距；倾斜；形状；位置<br>[32466](https://www.pixcakeai.com/guide/32466) | 全部左右解锁；眉距向右缩小，位置向右上移 | 部分：browLength/Width/Thickness/Spacing/Peak/Tail；无完整头尾长度、倾斜、位置等价控件 | 眉毛沿毛流及中心线变形，眼睑作为独立保护界面；各控制对中心线的作用分离 | 眉形确实变化、眼线不弯、毛流不断、低强度连续；两侧参数互不污染 | `brow.length / thickness / spacing / angle / shape / position` |

### 眼部几何

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-012 整体大小；眼高；眼宽；眼距；倾斜<br>[32466](https://www.pixcakeai.com/guide/32466) | 全部左右解锁 | 部分：eyeSize/Height/Width/Spacing/Tilt，geometry.rs SVF；审美未通过全量8分 | 眼睑曲线、眼球/虹膜与周围皮肤分层，尺寸不等同于矩形缩放 | 眼球不椭圆化、眼睑连续、旋转保局部面积、睫毛不拖影 | `eye.size / aspect / spacing / angle` |
| PCA-013 内眼角；外眼角；眼睑下至；位置；眼瞳大小<br>[32466](https://www.pixcakeai.com/guide/32466) | 左右独立；位置双向；眼瞳与整体眼睛大小分开 | 部分：eyeInnerCorner/OuterCorner/eyelid；眼瞳大小与眼睛位置独立能力缺失 | 内外眼角局部切线控制，下眼睑单独曲线；虹膜投影和眼白可见性模型 | 上眼睑不跟下至拉动；虹膜圆度/瞳孔中心合理；眼白露出无重复纹理 | `eye.corner.inner / outer / lowerLid / position / iris.size` |

### 鼻部几何

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-014 鼻大小；鼻高位置；左右位置；鼻梁；鼻翼；鼻尖<br>[32466](https://www.pixcakeai.com/guide/32466) | 鼻高在手册中指上下位置，不是三维隆鼻 | 部分：noseBridge/Wings/Length/Tip/Straight 等；鼻大小/位置不是一一对应 | 鼻部轮廓与鼻孔拓扑分离控制；立体高低必须独立于2D位置 | 鼻翼不越过鼻孔；鼻长不冒充鼻高位置；嘴唇和眼睛稳定 | `nose.size / position / bridge / wings / tip` |

### 唇部几何

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-015 嘴大小；嘴宽；上下/左右；倾斜；M唇；丰上唇；丰下唇；露龈<br>[32466](https://www.pixcakeai.com/guide/32466) [32469](https://www.pixcakeai.com/guide/32469) | 露龈左右；上下方向与眼睛相反；倾斜原文左右描述有重复歧义 | 部分：mouthWidth/Height、upperLip/lowerLip/lipPeak/lipCorner/philtrum；移动、露龈缺独立实现 | 嘴内外缘、上/下唇、牙龈分别建模；M唇不能只压一个椭圆；口周共享附着 | 闭嘴/露齿/大笑三类；牙齿不染色不拉裂；唇峰/角度需可测方向 | `mouth.size / position / angle / lip.shape / gingiva.cover` |

### 表情

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-016 AI完美表情；闭眼修复<br>[32468](https://www.pixcakeai.com/guide/32468) [TsKGUrCisN](https://www.pixcakeai.com/guide/TsKGUrCisN) | 同场次自动或手选人脸替换；闭眼要求同一人的睁眼参考图 | 缺口：没有跨照片同人表情替换 | 同人参考检索、姿态/光照对齐、局部重建；结果保存为独立候选 | 身份、视线、两眼一致；匹配失败不可换成别人的眼睛 | `expression.replace / eye.restoreFromReference` |
| PCA-017 露齿笑；抿嘴笑；闭合嘴巴；微笑嘴角<br>[32468](https://www.pixcakeai.com/guide/32468) | 露齿五档、闭嘴三档且可擦回；嘴角单独强度 | 部分：lipCorner 有局部形变，不等同笑容生成 | 保持原牙/唇细节的条件表情编辑；缺失牙齿生成只进入可选AI分支 | 牙数/边缘、口腔深度、法令纹联动；不开口时不得生成牙齿 | `expression.smile / mouth.close / mouth.corner` |

### 牙齿

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-018 牙齿祛瑕疵；边缘补齐；美白明度/祛黄；排齐；修复<br>[32469](https://www.pixcakeai.com/guide/32469) | 成人排齐五档；牙齿修复是替换牙列；需区别真实保留和生成 | 部分：teethWhiten；无独立亮度/祛黄控制、牙齿检测细分及修复 | 牙釉质mask、色度中和、牙边缘曲线；排齐与新牙生成单独算法 | 牙釉质纹理/齿缝阴影保留，口红/牙龈无变化；补齐不产生多牙 | `teeth.whiten / clean / edge.restore / align / replace` |

### 眼睛增强

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-019 眼球大小/水平/垂直；直视镜头；水平对称/大小对称与参考眼<br>[32470](https://www.pixcakeai.com/guide/32470) | 视线纠正单独于眼睛形变；大小对称可选参考侧 | 缺口：现有symmetry不是眼球视线重定向 | 虹膜/巩膜/眼睑分割+凝视估计；保持遮挡边界的局部重建 | 双眼凝视一致；高光随球面合理；眼镜遮挡及无虹膜证据拒绝 | `gaze.retarget / eye.symmetry` |
| PCA-020 亮眼；眼瞳亮度；眼白亮度；眼睛高光；祛红血丝；眼白净化；祛红眼<br>[32470](https://www.pixcakeai.com/guide/32470) | 前三种亮度支持左右，红血丝与红眼为不同问题 | 缺口：color.basic蒙版提亮只是手工底座 | 眼白结构保留与血管细线检测；虹膜纹理增强；红眼按瞳孔反光修正 | 不漂白巩膜阴影；不抹虹膜纹理；正常棕红虹膜负控 | `eye.brighten / sclera.clean / iris.redEye.correct` |
| PCA-021 祛双眼皮贴；眼镜祛反光；双/单眼皮效果<br>[32470](https://www.pixcakeai.com/guide/32470) | 贴膜左右；反光可擦回；眼皮程度、头尾高度/长度左右独立 | 缺口：eyelid形变不能替代贴膜修复/眼皮材质生成 | 结构及遮挡检测+局部参考/条件补全；眼皮褶皱沿真实眼睑曲线生成 | 睫毛、镜框、玻璃透射保留；无参考眼区不得声称真实还原 | `eye.adhesive.remove / glasses.reflection.remove / eyelid.crease` |
| PCA-022 眼神光：样式、程度、水平/垂直、旋转、翻转<br>[32470](https://www.pixcakeai.com/guide/32470) | 各参数支持左右；位置-100到100，旋转可达180度 | 缺口：无眼球高光贴图/光源控制 | 虹膜空间贴图与遮挡掩码、球面映射及原高光合成 | 高光不落在眼皮/眼白外；双眼光源方位一致；暗眼仍保留细节 | `eye.catchlight.adjust` |

### 祛瑕疵

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-023 祛痘；祛斑；质感保留；祛痣<br>[32464](https://www.pixcakeai.com/guide/32464) | 痘、斑可从完全去除往回保留；痣含脸和身体 | 部分：smooth启发式异常抑制及手工repair.heal；无独立痘/斑/痣识别 | 病灶/色斑类型分割+用户保留标记；健康纹理供体或局部修复；保留毛孔 | 分别测痘/斑召回与痣/雀斑误伤；圈外不变；不能磨糊一脸算去痘 | `skin.blemish.remove(type, preserve)` |
| PCA-024 祛油光；祛汗珠；祛鼻孔瑕疵；唇纹修整<br>[32464](https://www.pixcakeai.com/guide/32464) | 汗珠/干皮/鼻毛/鼻翼红为不同材料；需保留结构高光 | 缺口：无这些专用检测/修复 | 高光-反射分解、细线/水滴局部检测；按皮肤/鼻腔/唇部选择供体 | 高光压制不灰脸，汗珠消失且毛孔在，唇纹减弱不抹唇边 | `skin.specular.reduce / sweat.remove / nose.clean / lip.texture.repair` |
| PCA-025 祛黑眼圈；祛眼袋；保留卧蚕<br>[32464](https://www.pixcakeai.com/guide/32464) | 眼袋与黑眼圈分开；卧蚕可从默认保留逐步减弱 | 部分：darkCircle色度/亮度校正；eyeBag与tearTrough非零拒绝 | 分离色素、血管色偏与结构阴影；眼袋形态/光照需独立模型 | 眼下亮度匹配而卧蚕不消失；侧光不能错误抹平；严重眼袋不伪完成 | `underEye.darkCircle / bag / aegyoSal` |
| PCA-026 祛双下巴；阴影程度；下颌线增强<br>[32464](https://www.pixcakeai.com/guide/32464) | 双下巴有几何与阴影子控制，增强轮廓为光影操作 | 缺口：doubleChin非零拒绝，不能拿chin代替 | 脸颈共享界面、颈部遮挡与旧阴影移除；结构光照单独调整 | 脖子不截断、旧下颌阴影无双线、衣领/耳饰连贯 | `chin.double.reduce / jaw.shadow.enhance` |
| PCA-027 祛胡须：胡青/胡渣；妆容保护；胡型保护<br>[32464](https://www.pixcakeai.com/guide/32464) | 胡青与胡渣可独立保留；保护影响瑕疵/双下巴等操作 | 缺口：无胡须专项检测与材质处理 | 皮肤色素与毛发方向分离，保护mask进入每个受影响算法 | 胡型外缘、睫毛及妆容装饰不误删；去色不能等价去毛 | `beard.stubble.remove / beard.tint.reduce / portrait.protect` |
| PCA-028 祛抬头纹；川字纹；眼周纹；鼻背纹；法令纹；脸颊纹；木偶纹；嘴周纹；颈纹<br>[32464](https://www.pixcakeai.com/guide/32464) | 法令纹支持左右；各区域独立 | 缺口：通用smooth无独立纹路结构控制 | 沿纹理方向区分皱褶与毛孔，照明平整+必要的局部结构修复 | 每种纹路独立fixture；表情折线和自然阴影保留；年龄特征可控 | `skin.wrinkle.reduce(region, side)` |
| PCA-029 身体祛瑕疵；儿童/婴幼儿干皮；副乳；妊娠纹/线；纹身<br>[32464](https://www.pixcakeai.com/guide/32464) | 纹身可加减mask并更新；妊娠线可保留 | 缺口：已有手工修复但非自动专项 | 身体皮肤实例分割、病灶/纹身mask；结构赘肉与色素分支处理 | 大面积肤色一致；纹身选区外不变；褶皱与关节负控 | `body.skin.clean / stretchMark.remove / tattoo.remove` |

### 皮肤

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-030 中性灰平整；中性灰立体（五官/轮廓）；质感磨皮<br>[32465](https://www.pixcakeai.com/guide/32465) | 光照平整与骨相塑光分开 | 部分：skin.retouch smooth引导基底/纹理；无真正语义塑光 | 保纹理的照明/反射近似分离、结构光照约束；先小强度确定性版 | 毛孔频谱、肤色漂移、光照连续；侧光骨相不能被压扁 | `skin.tone.even / portrait.dodgeBurn` |
| PCA-031 高低频磨皮：脸/身、高频增强、低频平整；水润磨皮：脸/身<br>[32465](https://www.pixcakeai.com/guide/32465) | 主强度与高低频分开；水润是另一外观 | 部分：smooth/texture有相关数学但无完整脸身/高低频独立参数 | 按成像尺度分离细节和照明，skin mask排除五官，身体单独目标 | 纹理度与平整度响应独立；大图/缩图效果一致；不渗头发 | `skin.frequency.retouch / skin.hydrated.finish` |
| PCA-032 面部丰盈：全脸0–4档、额头、面中、面颊、鼻基底、眼窝、眉弓、下巴、嘴角<br>[32465](https://www.pixcakeai.com/guide/32465) | 部分部位分平整/提亮；眼窝左右、眼头开关 | 缺口：局部亮度或noseBase几何不是丰盈系统 | 分别估计凹陷光影和真实形态；控制光照补偿、结构改变的比例 | 提亮不代替体积；眼窝不能漂白眼皮；保留原照光源方向 | `face.volume.fill(region, side)` |
| PCA-033 锁骨增强；皮肤质感：哑光/缎面/水光、光泽/纹理/清晰度<br>[32465](https://www.pixcakeai.com/guide/32465) | 锁骨仅女/长辈女；光泽纹理双向 | 缺口：texture只是磨皮内纹理权重，非皮肤材质预设 | 锁骨结构阴影；皮肤高光、漫反射、微纹理分别建模 | 哑光不变灰、水光不涂油；保留噪声与毛孔区分；锁骨方向正确 | `body.clavicle.enhance / skin.material.adjust` |

### 肤色

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-034 肤色透亮；美白；红润；肤色选择8种；色温；色调<br>[32465](https://www.pixcakeai.com/guide/32465) | 脸身目标；色温冷暖双向，色调绿/紫双向 | 部分：whiten/warmth/redness；自动skin限脸部解析及可见颈，身体需手工圈选；无肤色tint/八预设及独立透亮 | 受肤色范围约束的亮度/色度变换，避免裁切；模型仅定位，可手工mask | 暗亮肤色都保色阶；肤色边缘无色环；眼白/嘴唇不变 | `skin.color.adjust` |
| PCA-035 AI多人肤色统一；AI全身肤色统一；脸部/身体局部统一；身体跟脸统一<br>[32465](https://www.pixcakeai.com/guide/32465) | 多人参考最亮脸；全身可吸管；脸、身体、多人是不同任务 | 缺口：现有redness不是多目标匹配 | 跨实例参考显式选择、肤色反射/光照解耦、低频色差拟合；保持各人自然肤色差异的可调目标 | 阴影不漂白，深肤色不强迫白化；脸颈手臂接缝平滑；参考人保持 | `skin.color.match(reference, targets)` |

### 妆容

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-036 高光立体；阴影立体；眉毛增强；眼妆增强；唇妆增强<br>[32471](https://www.pixcakeai.com/guide/32471) | 眉/眼左右；眼妆与唇妆分饱和/明度/立体 | 部分：browColor、lipColor/Brightness/Saturation；无眼妆/结构塑光 | 材质与皮肤分离后增强现有妆容，眉按毛流覆盖，唇按漫反射/高光 | 唇纹与高光保留，牙不染色，眼皮纹理不被纯色覆盖 | `makeup.enhance(region, side, component)` |
| PCA-037 妆容减淡：全脸；口红；修容高光/阴影；眉；卧蚕/美瞳/睫毛/眼线/眼影；腮红；面饰<br>[32471](https://www.pixcakeai.com/guide/32471) | 全脸与细分控制并存 | 缺口：负向browColor/lipSaturation不等于卸妆 | 可逆材质层估计；遮盖皮肤需要局部重建；独立保护五官几何 | 淡妆不改变身份，眼线/睫毛区分，装饰去除不留黑边 | `makeup.reduce(component)` |
| PCA-038 主题妆容；局部眉毛14种；眼妆8种；眼线6种；睫毛14种；美瞳11种<br>[32471](https://www.pixcakeai.com/guide/32471) | 主题含女性/男性/长辈/儿童风格；眉/眼线左右，睫毛还分上下 | 缺口：无主题及自动贴合材质层；手工paint不等价 | 可控素材参数化、曲线贴合、遮挡与镜面高光合成，主题仅组合 | 不同眼型/角度贴合，镜框/发丝遮挡正确，上下睫毛无越界 | `makeup.theme.apply / brow/eyeliner/lash/contactLens.apply` |
| PCA-039 卧蚕9种；腮红12种；口红13种；修容7种；雀斑5种；痣3种<br>[32471](https://www.pixcakeai.com/guide/32471) | 卧蚕可位置/大小；口红缎面/哑光/奶油/珠光/水光材质 | 缺口：当前唇色不是材质口红，其他专项缺失 | 素材位姿和皮肤/唇材料融合；修容依原光照，雀斑顺皮肤透视 | 边缘/毛孔/反光自然；唇部各材质有可区分质感；不复制造成贴纸感 | `makeup.aegyoSal / blush / lipstick / contour / decoration` |
| PCA-040 眉毛/眼线/睫毛/腮红/口红换色；RGB/HEX/吸管；收藏6色<br>[32471](https://www.pixcakeai.com/guide/32471) | 选色与材质程度分开 | 部分：paint.brush颜色是手工笔刷；自动妆容颜色协议缺口 | 统一线性色彩合成与可重放颜色值，输出色域约束 | 相同颜色在不同亮度下保留材质，不按整块替色 | `makeup.color.set` |

### 头发

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-041 发缝填充；额角/发际线/鬓角稀疏补发<br>[32472](https://www.pixcakeai.com/guide/32472) | 不同稀疏位置共同覆盖 | 缺口：hair mask只用于保护，非补发算法 | 头发方向场、局部可见供体与条件补全；保留真实发际线形态 | 根部方向连续、无重复梳状纹、不能覆盖眉毛与额饰 | `hair.density.fill` |
| PCA-042 颅顶/头顶；两侧发量；发际线<br>[32472](https://www.pixcakeai.com/guide/32472) | 颅顶、发量双向，发量左右独立；发际线上下 | 缺口：整体脸型实验不等于专用发型控制 | 整体头部材料与背景解耦，头发轮廓联合连续场，未知背景单独补全 | 发丝不拉成粗条；头饰刚性合理、耳朵不脱离；背景固定 | `hair.crown.height / hair.volume / hairline.position` |
| PCA-043 祛碎发：人物/背景；柔顺；内部发丝；光泽；祛白发<br>[32472](https://www.pixcakeai.com/guide/32472) | 碎发分主体内外；白发/碎发可擦回 | 缺口：无专用检测/颜色及纤维修复 | 发丝细线与遮挡分类，背景用小区修复，发内沿流向修复；镜面光泽单独调 | 保留自然碎发选择、耳饰不误删、白发与反光区分 | `hair.flyaway.remove / smooth / gloss / gray.recolor` |
| PCA-044 发色统一；13款发色选择<br>[32472](https://www.pixcakeai.com/guide/32472) | 统一双向分别参考发尾/发根，发色可调强度 | 缺口：通用HSL可手工调色但不等于自动发色统一 | 毛发alpha与漫反射色度分离，选择根/尾参考渐变匹配 | 透明发丝无染背景，头皮不过染，高光保留，分段染色可控 | `hair.color.match / hair.color.apply` |

### 身体

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-045 AI塑形；身型流畅；瘦身<br>[32473](https://www.pixcakeai.com/guide/32473) | AI塑形双向；流畅仅AI塑形正值1–100开启；瘦身独立 | 部分：body.adjust九项局部粗细；无自动体型/流畅整体方案 | 按骨轴/姿态和可见衣物轮廓构建区域目标，关节/接触/背景保护 | 裸体线条与衣物版型分开评；站姿/坐姿/交叠；非编辑肢体保持 | `body.shape.preset / body.contour.smooth / body.slim` |
| PCA-046 肩宽；AI瘦手臂/普通瘦手臂；瘦腰；瘦腿/大腿/小腿<br>[32473](https://www.pixcakeai.com/guide/32473) | 肩宽左右独立且负值收窄/正值拓宽；AI手臂左右；腿粗细双向分大/小腿；腰内收 | 部分：shoulder/arms/forearms/waist/belly/legs/thighs/calves；body.rs的shoulder正值收窄，与手册方向相反，须显式映射 | 实际部位alpha×骨轴分区，每段不同横截面曲线；接触处共享界面 | 手臂不扭躯干，膝踝连续，宽袖不冒充细臂，门框不拉弯 | `body.shoulder.width / arm.slim / waist.slim / leg.slim` |
| PCA-047 增高；长腿/大腿长/小腿长；腰长；上身长<br>[32473](https://www.pixcakeai.com/guide/32473) | 按结构位置改变长度，非全图纵向缩放 | 缺口：height/legLength非零拒绝，腰长/上身长无实现 | 轴向关节重定位与局部比例重建，固定脚底接触或显式调整构图 | 膝/踝/鞋底连续，站立地面不漂移，服装图案纵横比合理 | `body.height / leg.length / waist.length / torso.length` |
| PCA-048 小头；脖长；瘦脖；天鹅颈；直角肩<br>[32473](https://www.pixcakeai.com/guide/32473) | 脖粗细、天鹅颈、直角肩左右独立 | 缺口：headRatio/neck/neckLength/trapezius/shoulderLevel非零拒绝 | 头发/耳/颈/衣领共享附着，颈肩解剖线与光照联合 | 头不漂浮，发丝及衣领接缝无剪切，肩颈姿态不伪造 | `body.head.scale / neck.length / neck.width / shoulder.angle` |
| PCA-049 丰胸；胸位提升；美臀<br>[32473](https://www.pixcakeai.com/guide/32473) | 女性限制；胸大小手册向右收缩；美臀为膨胀 | 缺口：hips仅胯宽，不是胸部/臀部立体重塑 | 体表与衣物联合轮廓/纹理控制，胸臀位置独立于大小 | 胸衣/纽扣/臀缝连续，侧面体积和正面轮廓分别验收 | `body.chest.size / chest.position / hips.volume` |

### 手脚

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-050 手部中性灰；手/臂祛青筋；手宽/手指/指节美型；脚背祛青筋<br>[46608](https://www.pixcakeai.com/guide/46608) [9XYcMCvHXW](https://www.pixcakeai.com/guide/9XYcMCvHXW) | 脚青筋强度逐级且可擦回 | 缺口：手脚当前通常保护，暂无专用算法 | 手部骨架+指轮廓实例，皮肤血管与结构阴影分离；脚背专用skin区域 | 手指根部/指甲/戒指不断裂；静脉减弱但不抹腱骨纹理 | `hand.skin.retouch / hand.shape / foot.vein.reduce` |

### 衣物

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-051 衣服祛细褶皱；祛粗褶皱；上下装独立/联动<br>[32477](https://www.pixcakeai.com/guide/32477) | 可按所有人/人群，粗细强度独立 | 部分：clothing.adjust.smooth/texture 只做细纹频率分离；无粗褶专用 | 织物纹理、印花、缝线与褶皱光照分离；细折痕与大折面分别建模 | 细纹去除不抹布纹；粗褶明暗/结构一致；条纹/文字/纽扣负控 | `clothing.wrinkle.reduce(scale, garment)` |
| PCA-052 衣服祛瑕疵；祛汗渍；汗渍mask加减与更新<br>[32477](https://www.pixcakeai.com/guide/32477) | 碎屑/毛球和大块汗渍分开 | 缺口：通用heal/inpaint是手工底座，非专用检测 | 纤维背景上的异常点线检测；汗渍用颜色/照明分解和纹理保留补全 | 汗渍消失而衣物底色不换，刺绣/图案不删，边缘无接缝 | `clothing.debris.remove / sweat.remove` |
| PCA-053 边缘平整；边缘恢复；道具保护；擦回原细节<br>[32477](https://www.pixcakeai.com/guide/32477) | 衣物轮廓整理可回退，接触道具减弱形变 | 缺口：edges/fit等非零拒绝 | 衣物实例轮廓曲率约束+接触图，包/手/首饰作为附着对象而非矩形冻结 | 衣边顺滑但裁剪造型保留；手提物不断开；恢复层可精确擦回 | `clothing.edge.smooth / restore / protectContact` |
| PCA-054 衣服商品精修；鞋子祛污；裙子蓬松；裙摆平整；证件照换衣<br>[230950](https://www.pixcakeai.com/guide/230950) [qrjMSGJ2EJ](https://www.pixcakeai.com/guide/qrjMSGJ2EJ) [dUZWMMNsh0](https://www.pixcakeai.com/guide/dUZWMMNsh0) [jfMtohpr3O](https://www.pixcakeai.com/guide/jfMtohpr3O) [165437](https://www.pixcakeai.com/guide/165437) | 商品正/反与强/温和及2K/4K；鞋最多5双、强度/影调；裙3候选；换衣单/双人 | 缺口：均无等价专用能力；clothing颜色/细纹不能替代 | 商品结构保持编辑、鞋材质清理、裙装轮廓条件生成、换装人体/衣领配准分别实现 | 品牌文字缝线不变，鞋带完整；裙摆无多层假边；换衣头颈正确 | `garment.product.retouch / shoes.clean / skirt.volume / skirt.edge / outfit.replace` |

### 背景

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-055 纯色背景祛穿帮；背景/地面祛瑕疵；光影均匀及明度；祛色彩断层<br>[32474](https://www.pixcakeai.com/guide/32474) | 纯色专用，对非纯色无效果；瑕疵增强mask可编辑 | 缺口：通用inpaint不能等价纯色检测和平整 | 背景分类+低维照明面拟合，瑕疵/条带检测，阴影和人物保护 | 保留真实地面接触阴影；不是一键涂平；非纯色输入应明确不可用 | `background.studio.clean / tone.even / banding.remove` |
| PCA-056 背景增强；效果擦回<br>[32474](https://www.pixcakeai.com/guide/32474) | 饱和/对比增强，原细节可恢复 | 部分：蒙版color.basic可手工完成，无自动背景功能 | 背景实例mask，局部调色和效果mask独立 | 人物与透明头发不染色；擦回不改变后续其它层 | `background.enhance / effect.erase` |
| PCA-057 证件照换背景：推荐色、自定义色、图片；填充/居中/位置/大小；保留范围；边缘；不透明度<br>[32475](https://www.pixcakeai.com/guide/32475) | 主体、关联物、非关联物三种范围；居中才可位移背景 | 缺口：解析蒙版存在，不等于完整换背景产品 | 物理alpha与实例归属分离，前景去污染，背景独立变换 | 发丝和半透明纱不黑白边，关联物真实保留，背景不覆盖人物 | `background.replace / subject.keepScope / matte.refine` |
| PCA-058 祛背景色反光；原图阴影；新增软/硬/剪影阴影<br>[32475](https://www.pixcakeai.com/guide/32475) | 阴影强度、混合模式；新阴影位置/距离/角度/虚化/渐变 | 缺口：无完整色溢出和投影模型 | 前景颜色去污染和背景分离；原投影mask保留，新投影依地面/光源几何 | 衣服原色不误删，接触阴影不漂浮，抠发边缘无色圈 | `subject.despill / shadow.preserve / shadow.add` |
| PCA-059 换天空：素材/位置/翻转/角度/边缘/渐变/色温/色调/饱和/亮度/虚化/透明度；AI换天空<br>[32476](https://www.pixcakeai.com/guide/32476) [3raG48FD8J](https://www.pixcakeai.com/guide/3raG48FD8J) | 常规参数式与生成式两套，AI要求足够天空区域 | 缺口：无天空/地平线专用替换 | 天空分割与树梢alpha、地平线透视匹配；生成版独立候选 | 树枝/建筑不缺失，地平线不穿帮，常规替换不改主体 | `sky.replace / sky.generate` |
| PCA-060 人景色彩匹配；水面反射；水效虚化<br>[32476](https://www.pixcakeai.com/guide/32476) | 人物/景物分别映射新天空色，水面单独 | 缺口：全图LUT不能替代光照适配 | 天空照明估计、主体/环境分别调色，水面反射几何/波纹低通 | 反射方向正确，原人物不换脸，光照匹配可独立关掉 | `sky.harmonize / water.reflectSky` |

### 修复

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-061 AI祛路人：检测、选区加减、2候选、蒙版密度/羽化<br>[158249](https://www.pixcakeai.com/guide/158249) | 自动区分主体/背景人物，可手改、重生成 | 部分：repair.inpaint 手工mask；LaMa双端已接入；无完整路人自动判定 | 实例检测+主角显式保留+接触/影子mask；结构条件补全；候选质量比较 | 主体不误删，路人影子/反光纳入，建筑线贯通，复杂案例保留失败 | `background.person.remove` |
| PCA-062 智能消除：快速/生成式、每笔后执行、插座/标识检测、主体及关联物保护；创成式填充<br>[165435](https://www.pixcakeai.com/guide/165435) [hxJhjM76da](https://www.pixcakeai.com/guide/hxJhjM76da) | 快速/生成用途不同，生成填充接收文字并给2候选 | 部分：exemplar/LaMa及mask保护；无提示词填充、自动标识检测 | 小洞确定性纹理修补，大洞条件补全；最终写域由Rust统一裁定 | 模型不修改mask外/保护区/alpha；大洞结构非仅无空白 | `object.remove / region.generate` |
| PCA-063 污点修复；修补；仿制；液化九工具；效果擦除；祛黑白边<br>[32482](https://www.pixcakeai.com/guide/32482) [32479](https://www.pixcakeai.com/guide/32479) [32481](https://www.pixcakeai.com/guide/32481) [AlCcjw85ql](https://www.pixcakeai.com/guide/AlCcjw85ql) | 液化大小/密度/压力/速率/固定边；祛边程度及mask更新 | 部分：heal/clone、5种液化mode与protectMask；strength仅手调力度，无设备压感、密度、速率、独立固定边缘、平滑/左推或专用祛边 | 手工工具独立于模型；保持可累计场与原图采样；祛边先分辨污染色与真实轮廓 | 取样不跨透明/保护，冻结可靠，平滑不模糊图像，祛边不侵蚀真发丝 | `repair.heal / clone / patch / liquify.stroke / edge.decontaminate` |
| PCA-064 AI面部重绘五档：祛瑕/美化/质感/重塑/重构<br>[205366](https://www.pixcakeai.com/guide/205366) | 逐人强度、极速/标准、最多5人 | 缺口：skin.retouch不是生成重绘 | 身份约束的局部条件编辑，按档明确允许结构改变；独立AI结果层 | 原五官保留与主动重构分别评分；不得把生成毛孔称为恢复真细节 | `face.generative.retouch` |
| PCA-065 AI祛体毛；AI瘦肚子；AI加腹肌<br>[69YZJo2P2X](https://www.pixcakeai.com/guide/69YZJo2P2X) [230949](https://www.pixcakeai.com/guide/230949) [HQL6QugdKe](https://www.pixcakeai.com/guide/HQL6QugdKe) | 体毛针对手臂，多人；瘦肚子3候选，腹肌1候选 | 缺口：body.belly只是二维轮廓，不是生成赘肉修复 | 皮肤细毛修复与体型/肌肉条件生成分开，主体身份和衣物保护 | 没有额外肢体，腹肌符合姿态/光照，不改变身体其它区域 | `body.hair.remove / belly.generate / muscle.enhance` |
| PCA-066 AI超清：全图/脸/身体/背景；老照破损/划痕/噪点/可选上色<br>[158251](https://www.pixcakeai.com/guide/158251) [225672](https://www.pixcakeai.com/guide/225672) | 超清分区强度，老照可关上色 | 缺口：sharpness/noise是确定性滤波，不等价重建 | 退化估计+保身份恢复，真实性与生成细节标明；老照损伤mask专项 | 小字/眼睛不造假纹理，纹理与噪声分离；脸部身份独立对照 | `image.restore / image.superResolve / oldPhoto.colorize` |
| PCA-067 补草地；沙滩美化质感/平整；道具修复<br>[152790](https://www.pixcakeai.com/guide/152790) [hHmVSQDymd](https://www.pixcakeai.com/guide/hHmVSQDymd) [165436](https://www.pixcakeai.com/guide/165436) | 草地mask/2候选/色调；沙滩两强度；同一道具选区结果一致 | 缺口：通用repair.inpaint可手选但无场景专项 | 场景材料分类、纹理方向和尺度约束、主体/阴影保护 | 沙纹草叶不重复成块；道具轮廓文字保持；同mask确定性按所选模型声明 | `scene.grass.restore / sand.clean / prop.repair` |

### 光影创意

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-068 自然光；AI补光强度/色温；柔化阴影；夕阳光<br>[242864](https://www.pixcakeai.com/guide/242864) [230955](https://www.pixcakeai.com/guide/230955) [230956](https://www.pixcakeai.com/guide/230956) [230957](https://www.pixcakeai.com/guide/230957) | 补光是新光源，阴影柔化保留结构阴影；夕阳含天空/水面氛围 | 缺口：曝光和temperature不能代替重照明 | 条件照明编辑，材料与照明分离，人物细节回贴需物理一致 | 鼻影方向/皮肤高光合理、阴影不把骨相抹平、身份纹理不漂移 | `lighting.natural / fill / shadow.soften / sunset` |
| PCA-069 发光：黑柔/白柔/辉光，强度/半径/阈值/曝光/色散/着色；镜头虚化：景深/焦点/5光斑/清晰与虚化涂抹<br>[211524](https://www.pixcakeai.com/guide/211524) [165438](https://www.pixcakeai.com/guide/165438) | 发光可查看生效区；虚化智能或点选焦点 | 部分：film.glow/halation；无完整光学及深度虚化 | 高光提取+尺度PSF；单目深度/分层散景与遮挡正确合成 | 边缘无发光黑圈；头发前后遮挡合理；不能全背景等强高斯模糊 | `light.glow / depth.bokeh` |

### 场景创意

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-070 AI布景；AI追色2.0追色追光/追场景<br>[165433](https://www.pixcakeai.com/guide/165433) [230951](https://www.pixcakeai.com/guide/230951) | 布景主体/关联物、4K/6K、光照融合；追色参考样片与不同生成模式 | 缺口：无生成场景/光照迁移；LUT不是同等能力 | 背景合成和局部重照明分开；参考驱动生成只在显式允许范围 | 人像与道具形状不变、光源统一；结构变化作为独立候选 | `scene.replace / scene.referenceEdit` |
| PCA-071 春樱；夏晴；秋银杏/枫叶；冬小雪/大雪/雪后；背景扩图<br>[242863](https://www.pixcakeai.com/guide/242863) [230952](https://www.pixcakeai.com/guide/230952) [230954](https://www.pixcakeai.com/guide/230954) [211088](https://www.pixcakeai.com/guide/211088) [uXo7tBdQwi](https://www.pixcakeai.com/guide/uXo7tBdQwi) | 季节改变场景元素；扩图1.2/1.5/2/3倍或自定义边距、3候选 | 缺口：全新生成能力；resize/crop不等于扩图 | 分割目标环境后条件生成；扩图只写新增区域并检查透视 | 人物不季节化重绘，边界结构连续；原图区域受保护 | `scene.season.edit / canvas.outpaint` |

### 宠物

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-072 轮廓外祛碎毛；亮眼；虹膜增强；眼神光样式/大小/位置/旋转/翻转<br>[4WvzDIBakP](https://www.pixcakeai.com/guide/4WvzDIBakP) [DaMvAbwxp4](https://www.pixcakeai.com/guide/DaMvAbwxp4) | 眼神光左右可独立 | 缺口：人体模型不支持宠物 | 物种适配眼睛/毛发分割，不复用人脸固定比例 | 毛发保自然，异色虹膜保留，瞳孔形状不人脸化 | `pet.fur.clean / pet.eye.enhance` |

### 调色

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-073 曝光；亮度；对比；高光；阴影；白黑；纹理；清晰；去雾；鲜艳；饱和；色温；色调<br>[32455](https://www.pixcakeai.com/guide/32455) | 双向连续，全图或蒙版 | 已有子集：color.basic 20项，color.rs；8-bit sRGB限制 | 保持共享Rust确定性变换；向更高精度管线迁移单独验收 | 零值恒等、灰阶单调、肤色/高饱和不爆、端服务一致 | `color.basic.adjust` |
| PCA-074 输入/输出色阶与RGB单通道；曲线参数/RGB/明度/单通道；黑白；HSL八色<br>[32455](https://www.pixcakeai.com/guide/32455) | 含吸管与参数曲线；输出色阶不是输入色阶 | 部分：blackPoint/whitePoint/gamma、RGB曲线、HSL；输出色阶/亮度曲线未完整 | 明确线性光/显示域，单调插值、亮度与色度分离；补齐独立参数 | 灰阶连续，曲线身份、通道互不污染，亮度曲线不改色相 | `color.levels / curves / hsl / monochrome` |
| PCA-075 配置文件；自动/AI白平衡；曝光校准；多图色彩统一；相机原色校准<br>[32455](https://www.pixcakeai.com/guide/32455) [32452](https://www.pixcakeai.com/guide/32452) | RAW专用设置与普通图不同；多图参考当前图 | 缺口：无RAW/ICC完整管线及AI校准/跨图匹配 | 保留确定性RGB底座，RAW/ICC依成熟色彩库；校准参数可查看可改 | 色卡ΔE、曝光细节、跨相机/光照数据集；不以单图观感替代色准 | `color.profile / whiteBalance.auto / exposure.auto / batch.match` |
| PCA-076 色彩编辑器色域/平滑/色相/饱和/亮度/均匀度；颜色分级；可选颜色相对/绝对<br>[32455](https://www.pixcakeai.com/guide/32455) | 可反相范围；高/中/低光分级，混合和平衡 | 部分：HSL分区是底座，非完整色域/分级/CMYK式可选色 | 软色域权重与边界平滑，分级保持亮度，选择性颜色明确数学定义 | 色界无断层、肤色保护、相邻色响应连续，单项可重复 | `color.range.edit / grading / selective` |
| PCA-077 AI降噪；锐化半径/细节；亮度降噪细节/对比；色噪细节/平滑<br>[32455](https://www.pixcakeai.com/guide/32455) | AI标准/强力/自动；多种噪声分开 | 部分：sharpness/noise；无专项神经降噪/分量控制 | 信号/噪声估计和亮色分离；边缘纹理约束；AI可选 | 噪声残差、细线/毛孔保留、无过锐白边，真实弱光盲评 | `detail.sharpen / denoise / chromaDenoise` |

### 滤镜

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-078 内置滤镜；CUBE导入；颗粒全局/阴中高光/大小/粗糙/色彩；AI色彩风格含富士NC/NN等<br>[32455](https://www.pixcakeai.com/guide/32455) [32452](https://www.pixcakeai.com/guide/32452) | 静态滤镜与逐图计算风格不同；强度可调 | 部分：filter.film14预设、LUT、grain强度/seed；非官方富士标定，细粒参数未齐 | 场景输入标准化、胶片色彩响应、亮度依赖颗粒；精确标定与风格模拟分开 | 多场景肤色/高光/阴影、颗粒分辨率一致；不声称复刻官方曲线 | `filter.film / lut / grain / adaptiveStyle` |

### 调色

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-079 AI追色1.0全局/局部；影调/色彩强度；精准控制生成可编辑参数；参考区域；极致/和谐<br>[32452](https://www.pixcakeai.com/guide/32452) | 0–200主子强度；局部参考对应区域；局部暂不批量 | 缺口：手工LUT/参数不是语义参考追色 | 语义区域对应、色度与亮度分开拟合，输出可编辑变换与置信度 | 不同场景不把草绿迁移到肤色；参考/目标区域明确；低置信度可改 | `color.matchReference(sourceRegion, targetRegion)` |

### 局部范围

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-080 主体/皮肤/五官/头发/衣物/饰物子蒙版；背景天空/草/地/岩/水/沙/山/建筑；宠物<br>[32454](https://www.pixcakeai.com/guide/32454) | 可多人物、多部位组合，重复mask会叠加 | 部分：MediaPipe/BiSeNet/FASHN与手工mask；缺全套场景/配饰子类 | 实例语义选择与物理alpha分开，记录空间/版本；分类不冒充抠图 | 区域IoU和边缘alpha分别测；饰物/头发/颈无错误归属 | `selection.semantic.create` |
| PCA-081 AI智能/自由/快速选区；画笔；线性/径向；颜色/亮度范围；加减交并/反相/羽化<br>[32454](https://www.pixcakeai.com/guide/32454) [158252](https://www.pixcakeai.com/guide/158252) | AI边缘平滑/扩缩/羽化，mask列表可编辑；画笔不批量同步 | 部分：矩形/椭圆/多边形/笔刷/线性/径向/raster及反相；颜色亮度范围及组合未完整 | 统一mask代数、像素与归一化单位明确；AI只提供候选 | 加减交并数学合同、羽化不改变范围权限、边缘采样一致 | `selection.create / combine / refine` |

### 几何与工具

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-082 裁剪自由/比例/像素/证件照；旋转翻转；透视引导线/水平垂直；镜头畸变/暗角<br>[32455](https://www.pixcakeai.com/guide/32455) [32478](https://www.pixcakeai.com/guide/32478) | AI裁剪按人脸/留白；RAW机内裁剪；AI透视/水平 | 部分：crop/rotate/flip/resize/perspective；无RAW镜头配置/自动构图 | 成熟投影/镜头模型与显式人工引导；自动只是建议参数 | 直线/比例正确、坐标重投影，旋转后后续人像检测一致 | `geometry.crop / rotate / perspective / lens.correct` |

### 编辑底座

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-083 图层混合/透明度/组/蒙版；局部效果擦回；预设替换/叠加；批量；后期调色<br>[127932](https://www.pixcakeai.com/guide/127932) [32481](https://www.pixcakeai.com/guide/32481) [32457](https://www.pixcakeai.com/guide/32457) [32458](https://www.pixcakeai.com/guide/32458) [32480](https://www.pixcakeai.com/guide/32480) [230985](https://www.pixcakeai.com/guide/230985) | 同类型预设覆盖、异类型叠加；局部后期调色在人像之后 | 部分：操作图层/蒙版/撤销/预设/共享引擎；无完整多素材组与竞品继承规则 | 原图+非破坏操作图、组合参数语义、几何依赖追踪，先确保效果可重放 | 重排/开关/撤销/导出一致；AI和手工调用同操作；不被陈旧mask破坏 | `operation.update / effect.mask / preset.apply / graph.replay` |

### 交互

| 对账项/公开功能 | 调节粒度与依赖 | 当前实现 | 我们的算法策略 | 验收重点 | 原子意图（拟议） |
|---|---|---|---|---|---|
| PCA-084 AI点修：智能/快速区域、局部对话、调色、美化、轮廓锚点<br>[MlI1Jk6qm6](https://www.pixcakeai.com/guide/MlI1Jk6qm6) | 由选区推荐可用功能；轮廓锚点直接拖动 | 缺口：有MCP通用操作桥，不是选区智能助手 | 先把区域与已验证原子算法绑定，再可选Agent解释意图；不用新增平台替代算法 | 同一选区手工与MCP等价，不可用功能明确报错，预览后可撤销 | `region.inspect / operation.propose` |

## 人物、左右与依赖的统一约定

1. **人物**：实例选择、属性组和“所有人”是三种作用域；不要把同一张图内的人物序号当作跨图身份。选择/检测应可人工纠正；模糊、被遮挡、缺少可见轮廓时返回原因。
2. **左右**：手册多数只说“左右”，没有给出像素坐标或人体解剖左右的协议。我们的 API 应明确定义 `anatomical_left/right`，UI 显示实际高亮区域；镜像图和画面左右必须验证，不能直接复制中文滑杆方向。
3. **控制值**：主滑杆、子滑杆、左右联动、材质选择、生成候选不能压成一个 `amount`。中性值、正负方向、单位和可用范围需逐项定义；旧参数可解析但未实现时明确拒绝。
4. **蒙版**：语义归属、物理alpha、允许修改域、保护域、效果擦回域分别存储。图像几何变化后，应重检测或重投影依赖的mask/landmarks；不能沿用旧坐标而默默变形背景。
5. **共享附着**：脸—耳—颈—发、眼睑—眼球、唇—牙、手—袖口、腿—裙摆并不遵守同一曲线。相邻物体不能靠“先静止叠回”解决所有接缝；接触结构需要明确约束，未知遮挡需要拒绝或生成分支。
6. **AI可选**：手工修复、手工蒙版、仿制、液化、调色必须独立运行；自动检测失效允许手工补信息。需要生成缺失内容的功能单独表明模型依赖，不能没有模型时换个模糊算法却仍报成功。
7. **MCP精细度**：先为每个有效算法定义操作、作用对象、输入证据、允许写域和结果诊断，再从同一 registry 暴露；本轮不开发工作流编排。现有通用桥能传参不等于所有新意图已可调用。

## 8分验收应怎样执行

“有变化”“没有报错”“单测通过”和“8分自然度”不是同一个结论。每项需保留独立证据：输入图及来源、检测结果、参数、源代码/模型/构建哈希、原尺寸结果、局部对比、性能、失败样本。建议先覆盖正负低/中/高强度、多人/遮挡/滚转、浅深肤色、硬软光、背景直线、饰物/眼镜/手持物；派生旋转图不重复计作独立原照。

- **几何合同**：目标的轮廓/中心线确实改变；无折叠、裂缝、重复轮廓；附着处连续；非目标/保护区按合同保持；组合与跨零响应连续。
- **材质合同**：毛孔、发丝、布纹、唇纹和金属高光有分别的结构指标与原尺寸人工核对；不能靠磨糊换取低接缝差值。
- **生成合同**：真实允许修改域外、保护区与alpha保持；内部检查结构、身份、光照、重复纹理和残留物；“填满洞”不算修好。
- **图层合同**：手工和MCP相同操作同输出；浏览器/服务端参数与坐标一致；撤销、禁用、透明度、重排、保存重开、导出都有效；模型执行位置不能静默切换。
- **质量判定**：逐功能盲评与独立复核，保留失败而非取最佳样本；目标8分需要样本覆盖和明确评分依据。本文不新增自评分，不把手册里的营销效果当作已达到的基准。
- **性能**：冷启动/首次检测/拖动热响应/最终导出分别记录，至少标清图片分辨率、硬件和浏览器。不能将稀疏求解耗时替代整条渲染延迟，亦不预许3–8ms或60fps。

## 推荐推进顺序

第一组继续解决现有脸型/五官/身形的实际失败，尤其整体脸型与耳颈附着、不同五官曲线、眼球保形、腰腿关节连续；保留最小回归和原尺寸证据。第二组在稳定皮肤mask和反射/照明控制上补齐自动瑕疵、肤色统一与局部材质妆容，衣物先细纹/纹理保护，再粗褶与轮廓。第三组完善去人自动选择、主体保护和复杂背景重建；生成类头发/换装/表情/场景作为独立能力推进。

这不是按目录同时铺满功能的开发承诺。每个原子功能都应经过算法、小样本复核、失败集、共享核心合同、UI/服务端验证后再进入可用目录；不能先挂空滑杆。最新论文/模型作为候选，需要同输入、同参数范围、同成本的实测胜出才替换已知基线。

## 手册描述中的歧义和边界

- 面部重塑“嘴巴倾斜”两方向文字存在重复，不能直接作为参数真值；需竞品实机或官方纠正后才能确定对标方向。
- 快捷键页表头与部分 Cmd/Ctrl 对应有疑似倒置；参考线章节与快捷键章节的键位也不一致。本项目不照抄冲突键位。
- “AI背景扩图”操作步骤中混入“AI老照片修复”入口文字，按页面主题记录功能，不把该句当真实依赖。
- AI追色2.0部分地方分别写“追光追色/追色追光”和“氛围迁移/追场景”；功能分为参考光色和场景重构，暂不认定不同名称是四个模型。
- 功能截图可能含正文未描述的额外滑杆。本次未查看图片/视频，未收录的媒体细节不宣称不存在；新增时应另记媒体证据。
- 本次对竞品只有公开正文证据，没有运行其客户端、没有比较效果图，也无法验证“无损、秒级、自然”等产品表述。

## 外部AI工具箱

工具箱总说明已完整阅读，但它只给第三方工具列表和通用操作，**不等于每个工具都有独立技术规格**。当前不把这些命名当作本项目已实现能力；同类命名也不重复算算法。结构化证据保存每一包的工具名称，统一标记“仅聚合页列名、具体模型/参数未公开”。

| 工具包 | 创作者 | 本次列名数量 | 对项目的意义 |
|---|---|---:|---|
| 刘锐AIGC | 刘锐 | 22 | 按人像/材料/光照/场景/商品分解需求；均需独立算法与验收，当前不接平台 |
| 花生国风 | 绘影 | 27 | 按人像/材料/光照/场景/商品分解需求；均需独立算法与验收，当前不接平台 |
| 幻象Cos | 绘影 | 30 | 按人像/材料/光照/场景/商品分解需求；均需独立算法与验收，当前不接平台 |
| 光影大师 | 安涂声 | 18 | 按人像/材料/光照/场景/商品分解需求；均需独立算法与验收，当前不接平台 |
| 电商美工 | 瞬境AI | 13 | 按人像/材料/光照/场景/商品分解需求；均需独立算法与验收，当前不接平台 |
| 影像百宝箱 | 星美视觉工坊 | 9 | 按人像/材料/光照/场景/商品分解需求；均需独立算法与验收，当前不接平台 |
| 影像优化 | ZoeyAi | 4 | 按人像/材料/光照/场景/商品分解需求；均需独立算法与验收，当前不接平台 |
| AIGC设计 | 饭盒 | 11 | 按人像/材料/光照/场景/商品分解需求；均需独立算法与验收，当前不接平台 |
| 全能电商 | 图图的AI世界 | 9 | 按人像/材料/光照/场景/商品分解需求；均需独立算法与验收，当前不接平台 |
| 专业生图 | 构像 | 4 | 按人像/材料/光照/场景/商品分解需求；均需独立算法与验收，当前不接平台 |
| 婚纱摄影 | 大韩数码桔子AI | 21 | 按人像/材料/光照/场景/商品分解需求；均需独立算法与验收，当前不接平台 |
| 萌宠拍档 | 桔子AI | 11 | 按人像/材料/光照/场景/商品分解需求；均需独立算法与验收，当前不接平台 |

这些包还覆盖电商产品/珠宝/数码精修、背景透视融合、多视角、风格转绘、姿势、宠物、创意元素等。它们是后续范围参考，不挤占当前人像基础算法的修复。完整名单见 JSON `thirdPartyToolPackages`。

## 桌面目录及正文状态（92个URL）

“正文已读”指实际读完该URL的正文；“同文已读”指该父目录正文SHA与已读子章节相同。均未逐一查看内嵌媒体。

| 目录路径 | URL/ID | 正文状态 | 实际文章标题（仅不同者） |
|---|---|---|---|
| 桌面版使用手册 | [32445](https://www.pixcakeai.com/guide/32445) | 同文已读（47238） | 工作区页面介绍 |
| 桌面版使用手册 / 快速了解像素蛋糕 | [32446](https://www.pixcakeai.com/guide/32446) | 同文已读（47238） | 工作区页面介绍 |
| 桌面版使用手册 / 快速了解像素蛋糕 / 工作区页面介绍 | [47238](https://www.pixcakeai.com/guide/47238) | 正文已读 | — |
| 桌面版使用手册 / 快速了解像素蛋糕 / 创建一个新项目 | [32448](https://www.pixcakeai.com/guide/32448) | 正文已读 | — |
| 桌面版使用手册 / 快速了解像素蛋糕 / 管理项目内图片 | [32447](https://www.pixcakeai.com/guide/32447) | 正文已读 | — |
| 桌面版使用手册 / 快速了解像素蛋糕 / 回收站 | [32449](https://www.pixcakeai.com/guide/32449) | 正文已读 | — |
| 桌面版使用手册 / 快速了解像素蛋糕 / 预览视图模式 | [32450](https://www.pixcakeai.com/guide/32450) | 正文已读 | — |
| 桌面版使用手册 / AI挑图 | [230984](https://www.pixcakeai.com/guide/230984) | 正文已读 | — |
| 桌面版使用手册 / AI点修 | [MlI1Jk6qm6](https://www.pixcakeai.com/guide/MlI1Jk6qm6) | 正文已读 | — |
| 桌面版使用手册 / 预设 | [47205](https://www.pixcakeai.com/guide/47205) | 同文已读（32457） | 预设类型 |
| 桌面版使用手册 / 预设 / 预设类型 | [32457](https://www.pixcakeai.com/guide/32457) | 正文已读 | — |
| 桌面版使用手册 / 预设 / 新建预设 | [32458](https://www.pixcakeai.com/guide/32458) | 正文已读 | — |
| 桌面版使用手册 / 预设 / 分享&导入预设 | [158978](https://www.pixcakeai.com/guide/158978) | 正文已读 | — |
| 桌面版使用手册 / 预设 / 管理预设 | [32459](https://www.pixcakeai.com/guide/32459) | 正文已读 | — |
| 桌面版使用手册 / 色彩调节 | [32451](https://www.pixcakeai.com/guide/32451) | 正文已读 | 色彩调节页面概述 |
| 桌面版使用手册 / 色彩调节 / 色彩调节页面概述 | [47246](https://www.pixcakeai.com/guide/47246) | 同文已读（32451） | — |
| 桌面版使用手册 / 色彩调节 / AI色彩 | [32452](https://www.pixcakeai.com/guide/32452) | 正文已读 | — |
| 桌面版使用手册 / 色彩调节 / 色彩调节 | [32455](https://www.pixcakeai.com/guide/32455) | 正文已读 | — |
| 桌面版使用手册 / 色彩调节 / 局部调色 | [32454](https://www.pixcakeai.com/guide/32454) | 正文已读 | — |
| 桌面版使用手册 / 色彩调节 / 后期调色 | [230985](https://www.pixcakeai.com/guide/230985) | 正文已读 | — |
| 桌面版使用手册 / 人像美化 | [47206](https://www.pixcakeai.com/guide/47206) | 同文已读（47248） | 人像美化页面概述 |
| 桌面版使用手册 / 人像美化 / 人像美化页面概述 | [47248](https://www.pixcakeai.com/guide/47248) | 正文已读 | — |
| 桌面版使用手册 / 人像美化 / 人物属性栏 | [47211](https://www.pixcakeai.com/guide/47211) | 正文已读 | — |
| 桌面版使用手册 / 人像美化 / 祛除瑕疵 | [32464](https://www.pixcakeai.com/guide/32464) | 正文已读 | — |
| 桌面版使用手册 / 人像美化 / 皮肤调整 | [32465](https://www.pixcakeai.com/guide/32465) | 正文已读 | — |
| 桌面版使用手册 / 人像美化 / 面部重塑 | [32466](https://www.pixcakeai.com/guide/32466) | 正文已读 | — |
| 桌面版使用手册 / 人像美化 / 表情管理 | [32468](https://www.pixcakeai.com/guide/32468) | 正文已读 | — |
| 桌面版使用手册 / 人像美化 / 牙齿美化 | [32469](https://www.pixcakeai.com/guide/32469) | 正文已读 | — |
| 桌面版使用手册 / 人像美化 / 眼睛增强 | [32470](https://www.pixcakeai.com/guide/32470) | 正文已读 | — |
| 桌面版使用手册 / 人像美化 / 妆容调整 | [32471](https://www.pixcakeai.com/guide/32471) | 正文已读 | — |
| 桌面版使用手册 / 人像美化 / 头发调整 | [32472](https://www.pixcakeai.com/guide/32472) | 正文已读 | — |
| 桌面版使用手册 / 人像美化 / 手部美化 | [46608](https://www.pixcakeai.com/guide/46608) | 正文已读 | — |
| 桌面版使用手册 / 人像美化 / 脚部美化 | [9XYcMCvHXW](https://www.pixcakeai.com/guide/9XYcMCvHXW) | 正文已读 | — |
| 桌面版使用手册 / 人像美化 / 全身美型 | [32473](https://www.pixcakeai.com/guide/32473) | 正文已读 | — |
| 桌面版使用手册 / 图像美化 | [47207](https://www.pixcakeai.com/guide/47207) | 同文已读（32474） | 背景调整 |
| 桌面版使用手册 / 图像美化 / 背景调整 | [32474](https://www.pixcakeai.com/guide/32474) | 正文已读 | — |
| 桌面版使用手册 / 图像美化 / 证件照换背景 | [32475](https://www.pixcakeai.com/guide/32475) | 正文已读 | — |
| 桌面版使用手册 / 图像美化 / 换天空 | [32476](https://www.pixcakeai.com/guide/32476) | 正文已读 | — |
| 桌面版使用手册 / 衣物美化 | [32477](https://www.pixcakeai.com/guide/32477) | 正文已读 | — |
| 桌面版使用手册 / 宠物美化 | [sh6LHCS0qN](https://www.pixcakeai.com/guide/sh6LHCS0qN) | 同文已读（4WvzDIBakP） | 宠物祛瑕疵 |
| 桌面版使用手册 / 宠物美化 / 宠物祛瑕疵 | [4WvzDIBakP](https://www.pixcakeai.com/guide/4WvzDIBakP) | 正文已读 | — |
| 桌面版使用手册 / 宠物美化 / 眼睛调整 | [DaMvAbwxp4](https://www.pixcakeai.com/guide/DaMvAbwxp4) | 正文已读 | — |
| 桌面版使用手册 / 裁剪旋转 | [32478](https://www.pixcakeai.com/guide/32478) | 正文已读 | — |
| 桌面版使用手册 / 历史记录 | [47209](https://www.pixcakeai.com/guide/47209) | 正文已读 | — |
| 桌面版使用手册 / 添加自定义工具栏 | [32485](https://www.pixcakeai.com/guide/32485) | 正文已读 | — |
| 桌面版使用手册 / 搜索功能 | [32456](https://www.pixcakeai.com/guide/32456) | 正文已读 | — |
| 桌面版使用手册 / 手动工具 | [64528](https://www.pixcakeai.com/guide/64528) | 同文已读（AlCcjw85ql） | 祛除黑白边 |
| 桌面版使用手册 / 手动工具 / 祛除黑白边 | [AlCcjw85ql](https://www.pixcakeai.com/guide/AlCcjw85ql) | 正文已读 | — |
| 桌面版使用手册 / 手动工具 / 参考线 | [165385](https://www.pixcakeai.com/guide/165385) | 正文已读 | — |
| 桌面版使用手册 / 手动工具 / 手动液化 | [32479](https://www.pixcakeai.com/guide/32479) | 正文已读 | — |
| 桌面版使用手册 / 手动工具 / 手动修复 | [32482](https://www.pixcakeai.com/guide/32482) | 正文已读 | — |
| 桌面版使用手册 / 手动工具 / 手动涂抹画笔 | [32481](https://www.pixcakeai.com/guide/32481) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 | [104947](https://www.pixcakeai.com/guide/104947) | 同文已读（127932） | 创意模块功能说明 |
| 桌面版使用手册 / 创意模块 / 创意模块功能说明 | [127932](https://www.pixcakeai.com/guide/127932) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 人像——AI面部重绘 | [205366](https://www.pixcakeai.com/guide/205366) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 人像——闭眼修复 | [TsKGUrCisN](https://www.pixcakeai.com/guide/TsKGUrCisN) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 人像——AI瘦肚子 | [230949](https://www.pixcakeai.com/guide/230949) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 人像——AI加腹肌 | [HQL6QugdKe](https://www.pixcakeai.com/guide/HQL6QugdKe) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 人像——证件照换衣服 | [165437](https://www.pixcakeai.com/guide/165437) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 消除——创成式填充 | [hxJhjM76da](https://www.pixcakeai.com/guide/hxJhjM76da) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 消除——AI祛路人 | [158249](https://www.pixcakeai.com/guide/158249) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 消除——智能消除 | [165435](https://www.pixcakeai.com/guide/165435) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 消除——AI祛体毛 | [69YZJo2P2X](https://www.pixcakeai.com/guide/69YZJo2P2X) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 修复——AI超清 | [158251](https://www.pixcakeai.com/guide/158251) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 修复——AI补草地 | [152790](https://www.pixcakeai.com/guide/152790) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 修复——沙滩美化 | [hHmVSQDymd](https://www.pixcakeai.com/guide/hHmVSQDymd) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 修复——智能道具修复 | [165436](https://www.pixcakeai.com/guide/165436) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 修复——AI老照片修复 | [225672](https://www.pixcakeai.com/guide/225672) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 修复——AI衣服精修 | [230950](https://www.pixcakeai.com/guide/230950) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 修复——鞋子祛污 | [qrjMSGJ2EJ](https://www.pixcakeai.com/guide/qrjMSGJ2EJ) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 修复——裙子蓬松 | [dUZWMMNsh0](https://www.pixcakeai.com/guide/dUZWMMNsh0) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 修复——裙摆平整 | [jfMtohpr3O](https://www.pixcakeai.com/guide/jfMtohpr3O) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 场景——AI追色2.0（风格迁移） | [230951](https://www.pixcakeai.com/guide/230951) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 场景——AI布景 | [165433](https://www.pixcakeai.com/guide/165433) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 场景——AI换天空 | [3raG48FD8J](https://www.pixcakeai.com/guide/3raG48FD8J) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 场景——春日樱花 | [242863](https://www.pixcakeai.com/guide/242863) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 场景——夏日晴空 | [230952](https://www.pixcakeai.com/guide/230952) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 场景——秋天落叶 | [230954](https://www.pixcakeai.com/guide/230954) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 场景——冬日雪景 | [211088](https://www.pixcakeai.com/guide/211088) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 光影——自然光 | [242864](https://www.pixcakeai.com/guide/242864) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 光影——AI补光 | [230955](https://www.pixcakeai.com/guide/230955) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 光影——柔化阴影 | [230956](https://www.pixcakeai.com/guide/230956) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 光影——夕阳光 | [230957](https://www.pixcakeai.com/guide/230957) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 光影——发光 | [211524](https://www.pixcakeai.com/guide/211524) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 光影——镜头虚化 | [165438](https://www.pixcakeai.com/guide/165438) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 创意玩法——AI背景扩图 | [uXo7tBdQwi](https://www.pixcakeai.com/guide/uXo7tBdQwi) | 正文已读 | — |
| 桌面版使用手册 / 创意模块 / 手动工具 | [158252](https://www.pixcakeai.com/guide/158252) | 正文已读 | — |
| 桌面版使用手册 / AI工具箱 | [230427](https://www.pixcakeai.com/guide/230427) | 正文已读 | — |
| 桌面版使用手册 / 创建虚拟副本 | [3YX3UbkTVp](https://www.pixcakeai.com/guide/3YX3UbkTVp) | 正文已读 | — |
| 桌面版使用手册 / 批量应用效果至其他图片 | [32480](https://www.pixcakeai.com/guide/32480) | 正文已读 | — |
| 桌面版使用手册 / 导出图片设置 | [32483](https://www.pixcakeai.com/guide/32483) | 正文已读 | — |
| 桌面版使用手册 / 快捷键指南 | [32486](https://www.pixcakeai.com/guide/32486) | 正文已读 | — |

## iPad、手机与其他栏目目录覆盖

以下仅目录，正文均未抓取/阅读；不能从同名功能推断移动端与桌面端完全等价。完整110个iPad/手机节点及其URL已写入JSON目录。

### iPad版使用手册（43节点，含根）

- iPad版使用手册：[32496](https://www.pixcakeai.com/guide/32496)，仅目录。
- 快速了解像素蛋糕iPad版：[64516](https://www.pixcakeai.com/guide/64516)，仅目录。
- 快速了解像素蛋糕iPad版 / 工作区界面介绍：[32498](https://www.pixcakeai.com/guide/32498)，仅目录。
- 快速了解像素蛋糕iPad版 / iPad支持机型：[32497](https://www.pixcakeai.com/guide/32497)，仅目录。
- 快速了解像素蛋糕iPad版 / 创建一个新项目：[32499](https://www.pixcakeai.com/guide/32499)，仅目录。
- 预设：[64517](https://www.pixcakeai.com/guide/64517)，仅目录。
- 预设 / 新建预设：[32500](https://www.pixcakeai.com/guide/32500)，仅目录。
- 预设 / 管理预设：[32501](https://www.pixcakeai.com/guide/32501)，仅目录。
- 预设 / 导入预设：[32502](https://www.pixcakeai.com/guide/32502)，仅目录。
- 色彩调节：[64518](https://www.pixcakeai.com/guide/64518)，仅目录。
- 色彩调节 / AI色彩：[81pN1Q3lMa](https://www.pixcakeai.com/guide/81pN1Q3lMa)，仅目录。
- 色彩调节 / 色彩调节：[32504](https://www.pixcakeai.com/guide/32504)，仅目录。
- 色彩调节 / 色彩调节功能释义：[32505](https://www.pixcakeai.com/guide/32505)，仅目录。
- 人像美化：[64519](https://www.pixcakeai.com/guide/64519)，仅目录。
- 人像美化 / 人物属性栏：[32506](https://www.pixcakeai.com/guide/32506)，仅目录。
- 人像美化 / 祛除瑕疵：[32507](https://www.pixcakeai.com/guide/32507)，仅目录。
- 人像美化 / 皮肤调整：[32508](https://www.pixcakeai.com/guide/32508)，仅目录。
- 人像美化 / 面部重塑：[32509](https://www.pixcakeai.com/guide/32509)，仅目录。
- 人像美化 / 表情管理：[32511](https://www.pixcakeai.com/guide/32511)，仅目录。
- 人像美化 / 牙齿美化：[32512](https://www.pixcakeai.com/guide/32512)，仅目录。
- 人像美化 / 眼睛增强：[32513](https://www.pixcakeai.com/guide/32513)，仅目录。
- 人像美化 / 妆容调整：[32514](https://www.pixcakeai.com/guide/32514)，仅目录。
- 人像美化 / 头发调整：[32515](https://www.pixcakeai.com/guide/32515)，仅目录。
- 人像美化 / 手部美化：[98564](https://www.pixcakeai.com/guide/98564)，仅目录。
- 人像美化 / 全身美型：[32516](https://www.pixcakeai.com/guide/32516)，仅目录。
- 图像美化：[64520](https://www.pixcakeai.com/guide/64520)，仅目录。
- 图像美化 / 背景调整：[32517](https://www.pixcakeai.com/guide/32517)，仅目录。
- 图像美化 / 证件照换背景：[32518](https://www.pixcakeai.com/guide/32518)，仅目录。
- 图像美化 / 换天空：[32519](https://www.pixcakeai.com/guide/32519)，仅目录。
- 衣服调整：[32520](https://www.pixcakeai.com/guide/32520)，仅目录。
- 裁剪旋转：[32521](https://www.pixcakeai.com/guide/32521)，仅目录。
- 手动工具：[64530](https://www.pixcakeai.com/guide/64530)，仅目录。
- 手动工具 / 手动液化：[32522](https://www.pixcakeai.com/guide/32522)，仅目录。
- 手动工具 / 手动修复：[32525](https://www.pixcakeai.com/guide/32525)，仅目录。
- 手动工具 / 手动涂抹画笔：[32524](https://www.pixcakeai.com/guide/32524)，仅目录。
- 创意模块：[UGnuheyhEA](https://www.pixcakeai.com/guide/UGnuheyhEA)，仅目录。
- 创意模块 / AI祛路人：[zcbO6shiuE](https://www.pixcakeai.com/guide/zcbO6shiuE)，仅目录。
- 创意模块 / AI换场景：[5FEXocjpnH](https://www.pixcakeai.com/guide/5FEXocjpnH)，仅目录。
- 创意模块 / AI超清：[ISHtYnbhxd](https://www.pixcakeai.com/guide/ISHtYnbhxd)，仅目录。
- 创意模块 / AI打光：[MZxl2tOdKM](https://www.pixcakeai.com/guide/MZxl2tOdKM)，仅目录。
- 批量应用效果至其他图片：[32523](https://www.pixcakeai.com/guide/32523)，仅目录。
- 导出图片设置：[32526](https://www.pixcakeai.com/guide/32526)，仅目录。
- 清理缓存：[32527](https://www.pixcakeai.com/guide/32527)，仅目录。
### 手机版使用手册（67节点，含根）

- 手机版使用手册：[166579](https://www.pixcakeai.com/guide/166579)，仅目录。
- iPhone版使用手册：[181169](https://www.pixcakeai.com/guide/181169)，仅目录。
- iPhone版使用手册 / 快速了解像素蛋糕iPhone版：[124117](https://www.pixcakeai.com/guide/124117)，仅目录。
- iPhone版使用手册 / 快速了解像素蛋糕iPhone版 / 工作区页面介绍：[124121](https://www.pixcakeai.com/guide/124121)，仅目录。
- iPhone版使用手册 / 快速了解像素蛋糕iPhone版 / 3步完成批量修图：[127940](https://www.pixcakeai.com/guide/127940)，仅目录。
- iPhone版使用手册 / 快速了解像素蛋糕iPhone版 / 开始修图：[124163](https://www.pixcakeai.com/guide/124163)，仅目录。
- iPhone版使用手册 / 快速了解像素蛋糕iPhone版 / 管理项目：[124175](https://www.pixcakeai.com/guide/124175)，仅目录。
- iPhone版使用手册 / 功能介绍：[124173](https://www.pixcakeai.com/guide/124173)，仅目录。
- iPhone版使用手册 / 功能介绍 / 相机/监视器模式：[225513](https://www.pixcakeai.com/guide/225513)，仅目录。
- iPhone版使用手册 / 功能介绍 / 精修页面概述：[124745](https://www.pixcakeai.com/guide/124745)，仅目录。
- iPhone版使用手册 / 功能介绍 / 预设：[124480](https://www.pixcakeai.com/guide/124480)，仅目录。
- iPhone版使用手册 / 功能介绍 / 色彩：[124481](https://www.pixcakeai.com/guide/124481)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像：[124482](https://www.pixcakeai.com/guide/124482)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像 / 人像页面概述：[125943](https://www.pixcakeai.com/guide/125943)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像 / 人像预设：[z7TfOGbLWn](https://www.pixcakeai.com/guide/z7TfOGbLWn)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像 / 祛除瑕疵：[125944](https://www.pixcakeai.com/guide/125944)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像 / 皮肤美化：[125945](https://www.pixcakeai.com/guide/125945)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像 / 肤色：[WCyNmANXho](https://www.pixcakeai.com/guide/WCyNmANXho)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像 / 面部重塑：[125956](https://www.pixcakeai.com/guide/125956)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像 / 身体塑形：[125979](https://www.pixcakeai.com/guide/125979)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像 / 美妆：[125976](https://www.pixcakeai.com/guide/125976)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像 / 面部丰盈：[aPfqjknJOw](https://www.pixcakeai.com/guide/aPfqjknJOw)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像 / 头发调整：[125978](https://www.pixcakeai.com/guide/125978)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像 / 眼部优化：[125975](https://www.pixcakeai.com/guide/125975)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像 / 牙齿美化：[125963](https://www.pixcakeai.com/guide/125963)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像 / 手部美化：[217308](https://www.pixcakeai.com/guide/217308)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像 / 表情管理：[189633](https://www.pixcakeai.com/guide/189633)，仅目录。
- iPhone版使用手册 / 功能介绍 / 人像 / 衣物美化：[3irOe55nFQ](https://www.pixcakeai.com/guide/3irOe55nFQ)，仅目录。
- iPhone版使用手册 / 功能介绍 / 构图：[166578](https://www.pixcakeai.com/guide/166578)，仅目录。
- iPhone版使用手册 / 功能介绍 / AI工具：[242728](https://www.pixcakeai.com/guide/242728)，仅目录。
- iPhone版使用手册 / 功能介绍 / 特效：[215699](https://www.pixcakeai.com/guide/215699)，仅目录。
- iPhone版使用手册 / 功能介绍 / 衣物：[166574](https://www.pixcakeai.com/guide/166574)，仅目录。
- iPhone版使用手册 / 功能介绍 / 液化：[124483](https://www.pixcakeai.com/guide/124483)，仅目录。
- iPhone版使用手册 / 功能介绍 / 消除：[189632](https://www.pixcakeai.com/guide/189632)，仅目录。
- iPhone版使用手册 / 功能介绍 / 背景：[177625](https://www.pixcakeai.com/guide/177625)，仅目录。
- iPhone版使用手册 / 功能介绍 / 去拼图：[217016](https://www.pixcakeai.com/guide/217016)，仅目录。
- iPhone版使用手册 / 功能介绍 / 批量应用效果至其他照片：[128379](https://www.pixcakeai.com/guide/128379)，仅目录。
- iPhone版使用手册 / 功能介绍 / 导出照片设置：[125995](https://www.pixcakeai.com/guide/125995)，仅目录。
- iPhone版使用手册 / 功能介绍 / 分享：[189634](https://www.pixcakeai.com/guide/189634)，仅目录。
- Android版使用手册：[181170](https://www.pixcakeai.com/guide/181170)，仅目录。
- Android版使用手册 / 快速了解像素蛋糕Android版：[174355](https://www.pixcakeai.com/guide/174355)，仅目录。
- Android版使用手册 / 快速了解像素蛋糕Android版 / 工作区页面介绍：[174356](https://www.pixcakeai.com/guide/174356)，仅目录。
- Android版使用手册 / 快速了解像素蛋糕Android版 / 3步完成批量修图：[180961](https://www.pixcakeai.com/guide/180961)，仅目录。
- Android版使用手册 / 快速了解像素蛋糕Android版 / 创建新项目：[175121](https://www.pixcakeai.com/guide/175121)，仅目录。
- Android版使用手册 / 快速了解像素蛋糕Android版 / 管理项目：[176677](https://www.pixcakeai.com/guide/176677)，仅目录。
- Android版使用手册 / 功能介绍：[175194](https://www.pixcakeai.com/guide/175194)，仅目录。
- Android版使用手册 / 功能介绍 / 精修页面概述：[175195](https://www.pixcakeai.com/guide/175195)，仅目录。
- Android版使用手册 / 功能介绍 / 预设：[175604](https://www.pixcakeai.com/guide/175604)，仅目录。
- Android版使用手册 / 功能介绍 / 人像：[175796](https://www.pixcakeai.com/guide/175796)，仅目录。
- Android版使用手册 / 功能介绍 / 人像 / 人像页面概述：[175808](https://www.pixcakeai.com/guide/175808)，仅目录。
- Android版使用手册 / 功能介绍 / 人像 / 祛除瑕疵：[202179](https://www.pixcakeai.com/guide/202179)，仅目录。
- Android版使用手册 / 功能介绍 / 人像 / 皮肤调整：[202664](https://www.pixcakeai.com/guide/202664)，仅目录。
- Android版使用手册 / 功能介绍 / 人像 / 面部重塑：[202665](https://www.pixcakeai.com/guide/202665)，仅目录。
- Android版使用手册 / 功能介绍 / 人像 / 表情管理：[yxWZwYRkdl](https://www.pixcakeai.com/guide/yxWZwYRkdl)，仅目录。
- Android版使用手册 / 功能介绍 / 人像 / 牙齿美化：[202666](https://www.pixcakeai.com/guide/202666)，仅目录。
- Android版使用手册 / 功能介绍 / 人像 / 眼睛增强：[202667](https://www.pixcakeai.com/guide/202667)，仅目录。
- Android版使用手册 / 功能介绍 / 人像 / 妆容调整：[202668](https://www.pixcakeai.com/guide/202668)，仅目录。
- Android版使用手册 / 功能介绍 / 人像 / 头发调整：[202669](https://www.pixcakeai.com/guide/202669)，仅目录。
- Android版使用手册 / 功能介绍 / 人像 / 手部美化：[5cxALIbnV4](https://www.pixcakeai.com/guide/5cxALIbnV4)，仅目录。
- Android版使用手册 / 功能介绍 / 人像 / 全身美型：[202670](https://www.pixcakeai.com/guide/202670)，仅目录。
- Android版使用手册 / 功能介绍 / 色彩：[175784](https://www.pixcakeai.com/guide/175784)，仅目录。
- Android版使用手册 / 功能介绍 / 构图：[185258](https://www.pixcakeai.com/guide/185258)，仅目录。
- Android版使用手册 / 功能介绍 / 衣物：[202672](https://www.pixcakeai.com/guide/202672)，仅目录。
- Android版使用手册 / 功能介绍 / 液化：[176319](https://www.pixcakeai.com/guide/176319)，仅目录。
- Android版使用手册 / 功能介绍 / 背景：[CdIvXGZ9Ng](https://www.pixcakeai.com/guide/CdIvXGZ9Ng)，仅目录。
- Android版使用手册 / 功能介绍 / 导出照片设置：[176321](https://www.pixcakeai.com/guide/176321)，仅目录。
- Android版使用手册 / 功能介绍 / 批量应用效果至其他照片：[176320](https://www.pixcakeai.com/guide/176320)，仅目录。

其他全站一级栏目：[像素助手](https://www.pixcakeai.com/guide/233147)、[联机拍摄](https://www.pixcakeai.com/guide/170676)、[像素产品协同工作流](https://www.pixcakeai.com/guide/52595)、[一体化智能工作流](https://www.pixcakeai.com/guide/166799)、[对接ERP](https://www.pixcakeai.com/guide/212685)、[常见问题](https://www.pixcakeai.com/guide/32541)、[视频教程](https://www.pixcakeai.com/guide/32570)、[更新日志](https://www.pixcakeai.com/guide/32578)、[会员体系](https://www.pixcakeai.com/guide/235346)、[积分商城](https://www.pixcakeai.com/guide/103791)。均只读取目录，未评审正文。
