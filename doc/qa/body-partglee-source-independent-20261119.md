# PartGLEE 原图部件语义：独立审查

2026-11-19。固定官方 RN50/PACO 配置、完整 531 类和无提示整图推理，在 `side-fashion` 粉色针织衫上提供了此前 SAM2 无提示候选库缺少的**衣身—弯曲前袖语义**。这是实质性的源证据进展，但尚未成为自动部件分配、光学分层或最终收腰效果通过。

第二张 `tilted-shirt` 在同一固定 PACO 任务下没有获得可靠袖—衣身区分。审计完整词表后确认其中只有 `sweater` 服装部件家族，没有 `shirt/blouse/jacket/coat/dress/skirt/pants`；不能把它对衬衣的失败扩大为所有开放词汇 PartGLEE 路线失败，也不能把针织衫单图成功推广到全部衣物。

## 实际实验边界

本地结果：`test/results/partglee-20261119/official-paco-v2/`。完整协议保存了源 SHA、全部原类、脚本与 CPU 适配哈希。每图保留全部 300 个 query 的原始 logits、masks、boxes；展示只按预先固定的每个原 `sweater` 类 top 3 排序，mask 显示阈值为原 logit 0。这是展示策略，不是边界真值或已定型的自动选取器。

预处理使用官方 `ResizeShortestEdge(800,max1333)`、RGB 归一化与 32 对齐，不裁眼/身体 ROI，不输入人工点、旧分层或编辑后图。执行 agent 报告 CPU 单图推理约 68.08 秒和 82.82 秒；本审查者没有重复执行模型。它是当前机器可运行的源准备候选，不是浏览器实时滑块方案。

此前可行性调查与引用见 [partglee-cpu-feasibility-20261119.md](../research/partglee-cpu-feasibility-20261119.md)。该调查当时尚未实际运行，不能用后续运行成功改写此前证据状态。

## Side-fashion：信息确实新增，边界仍须验证

审查者查看全部 24 个固定展示格，另按原图 `[480,205,705,425]` 查看 body/sleeve/yoke/shoulder 共 12 个原尺寸部件 crop，没有只看置信度。完整材料交界大形与独立核心检查相互支持：

| 原 query | 独立冻结衣身核心 | 独立冻结上袖核心 | 独立冻结前袖核心 | 原尺寸观察 |
|---|---:|---:|---:|---|
| body q100 | 100% | 0% | 0% | 后方衣身、翻折下缘与前袖分开；主体连续 |
| body q112 | 100% | 0% | 0% | 同样获得后方衣身，边缘与 q100 并非相同 |
| body q120 | 99.02% | 0% | 0% | 衣身有大面积缺损，不能因核心命中接受完整 mask |
| sleeve q104 | 0% | 98.28% | 100% | 弯曲上袖、前袖主体正确分开；肩顶另有部件类别，不能假定一个 mask 完整覆盖所有肩袖连接 |
| sleeve q113 | 0% | 100% | 100% | 肩根孔洞、衣身折边碎片误入；不能仅因 core 更完整就优先它 |
| sleeve q122 | 0% | 0% | 83.16% | 只有部分前臂，不能作整只袖 |

上述六个候选在冻结的独立外套、丹宁、手、包、背景核心中均为零。核心来自此前看 SAM2 候选前冻结的 `/tmp/body-independent-20261119/side-fashion/frozen-source-evaluation.json`，没有输入此次模型。复核结果保存在 `/tmp/body-independent-20261119/partglee/side-core-stats.json` 与 `side-parts.png`。

这些矩形是稀疏确定材料证据，不能代表全部边界。图片本身已经被本项目反复研究，不是全新照片盲测。肩、yoke、neckband 等部件还有重叠、遗漏和混分，不能任意相加后宣称真实运动归属；更不能把语义 mask 的 0/1 当作透明度。

相对于旧无提示 SAM2 库中衣身和袖共享完全相同成员向量的反例，q100/q104 已能提供不同身份信息。原信息不足的瓶颈因此有可验证的候选解法；自动选择、边界不确定区与效果验证仍未完成。

## Tilted-shirt：固定 PACO 任务未通过

完整 24 格固定展示中，`sweater:body` 首位 q120 主要指向下方裙子；另一个 body 候选 q172 主要落在横跨前袖，q142 只有胸前窄片。`sweater:sleeve` 首位 q105 混入衣身，另外两个主要是袖口/少量袖段，没有完整分离可用的胸衣与横前袖。不能将这些候选交给 matting，希望后者补齐材料身份。

`garment-vocabulary-audit.json` 枚举全部 75 个 object 类及 531 个 object/part 类。衣物关键词命中只包括 `sweater` 与其 body/cuff/hem/neckband/shoulder/sleeve/yoke；`chair:skirt` 是椅裙，并非服装裙。没有不存在的 `shirt` logits 列可从当前输出中导出。

因此保留**官方 PACO 任务在该衬衣样本失败**的结论。后续如测试自定义服装词表，应另立源实验协议，固定类别与选择规则后再推理，不能回改本轮词表再冒充原方案通过。该照片实际腰外缘也被衣袖遮挡，材料识别实验成功与否不改变“不能用它证明真实收腰轮廓”的可观察性限制。

## 下一步应回答的问题

优先验证自动源分配能否把 PartGLEE 的新增语义可靠转成运动角色，同时保住完整可见衣缘和前景袖：源身份只决定候选角色，光学边界另估计，几何不能跨静止衣物取半径。先固定针织类别适用范围和选择规则，再在新的适用照片上验证；保留失败与手动纠正路径。

目前没有自动修图已可用或整体达到 8 分的证据。新模型解决的是一个明确的信息缺口，不应再把一次源候选成功直接计为用户功能完成。

## 固定 custom 词表开发集（同源复核，非盲测）

为避免把 PACO 没有衬衣类别的失败误判为开放词汇模型失败，另运行了官方
`custom_object_categories` / `custom_part_categories` 分支。三张已反复研究过的
原图共用一套固定词表：对象 `shirt, blouse, sweater, jacket, coat, dress`；每个对象的
部件按 `body, sleeve, cuff, shoulder, hem` 拼成 `object:part`，共 36 类。没有为某张图
改名、补类别、改阈值或重复抽样，也没有把人工点、旧源层、Pose 或编辑图输入模型。

结果保存在 `test/results/partglee-20261119/custom-garments-v1/`，协议、每类 top3、
原始 logits/masks/boxes、内部 300 object/500 part 候选和 parent 来源全部保留。官方
custom 分支的可视化 top-k 表没有注册 custom，输出按官方 `VISUALIZE=True` 路径保留
50 object + 10 part；每类展示从完整内部候选固定排序得到，不能把展示截断当成真值。

当前开发集的预先检查结论：

| 原图 | 固定对象 top1 | 部件候选 | 判定 |
|---|---|---|---|
| fashion-standing（粉色男士衬衣） | `shirt`，约 0.297 | `shirt:body` q67 是中心衣身，`shirt:sleeve` q28 只覆盖部分袖区，完整袖缘仍需独立确认 | 有局部语义增益，暂不自动接受 |
| side-fashion（粉色针织衫） | `sweater`，约 0.380 | `sweater:body` q10、`sweater:sleeve` q427 与 PACO 成功方向一致；前袖候选较有用，肩根/后层仍需边界审查 | 只允许作为已审袖身份的源对照候选 |
| tilted-shirt（粉色内层/白色外衫） | `dress`，约 0.488 | body/sleeve 混入裙、外衫与内层，最高对象类别明显为 taxonomy 混淆 | 拒绝自动源分配 |

这些 score 是原始 sigmoid-logit 排序，未校准，不能单独决定材料身份。尤其 custom
分支将未训练过的对象—部件组合编码成文本并不等于有对应监督；`tilted-shirt` 的
`dress` 高分也不代表衣物类别正确。当前只把 side-fashion 已经由独立全图/原尺寸
审查通过的前袖 proposal 交给后续单变量光学对照，衣身、coat、手包和隐藏后层继续
沿用原角色保护；因此仍是**半自动源候选研究**，不是通用身形功能完成。

## Side-fashion q104 袖子 source-control 的独立有限通过

2026-11-19 追加复核。官方 PACO `sweater:sleeve` q104 被固定为唯一替换项，其他
手、包、衣身/裙、外套和后层角色保持原冻结控制；没有把 q100 衣身从另一父级合并进来，
也没有把编辑结果、人工点或旧候选输入 PartGLEE。结果目录为
`test/results/partglee-20261119/sleeve-source-control/`，本段只记录独立复核，未修改生成
报告中的哈希或 `qualityAccepted` 标志。

| 检查项 | 结果 |
|---|---|
| `+0` | 与 source RGBA 字节精确相同，changed=0 |
| `-30` / `+30` | changed=6875 / 6543；最大位移 5.37 / 5.03 px |
| 受保护和不应运动区域 | `changedOpaqueFrontPixels=0`、`changedProtectedPixels=0`、`changedZeroFieldPixels=0` |
| 独立视觉检查 | waist、denim-left、foreground-sleeve、coat-right 全图和放大 crop 未见新裂缝或阶梯；左侧可见丹宁边有连续双向响应，袖、手、包和右外套基本冻结 |

因此这组 ±30 只得到**有限的 source-layer 对照通过**：它证明 q104 在该已知
`side-fashion` 图上能提供比旧 SAM2 更有用的袖子身份，并在当前固定手工角色与
MEMatte 估计下产生可观察、连续的局部响应。它没有证明完整衣物、肩/yoke、遮挡顺序或
真实腰部轮廓已经正确。前景 matte 仍有 10 px trimap unknown，另有 16 px 封闭袖内
漏检被明确保留为 unknown；source geometry 仍是人工冻结身份，不能宣称透明度真值。

该结果的生成报告仍为 `visualAccepted=false`/`qualityAccepted=false`，这是有意保留的
审计状态。不得把这次有限通过计入整体可用性、自动泛化或 8/10 评分；±60 及未遮挡
解剖腰部尚未通过。
