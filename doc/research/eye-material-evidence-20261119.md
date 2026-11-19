# 眼球与细睫毛源身份：实现方向核查

2026-11-19。问题是彩色人像中，仍可见的虹膜暗纹、高光与随眼睑移动的细睫毛如何区分。需要源材料身份，不能用重建原图 RGB 或眼部平均分割分数替代。

本次为定向实施调研，不是系统综述或 SOTA 排名。检索 arXiv、Crossref 与 GitHub；查询与原始响应保存在本地 `test/results/eye-material-literature-20261119/`。主要查询为 `eyelash AND segmentation`，以及 2023-01-01 至检索日的标题 `eye segmentation / iris segmentation / eyelash`。GitHub 补充查询 `eyelash segmentation`、`eye segmentation iris pretrained`；宽泛眼部检索含大量鸟瞰视觉和医学无关结果，未据其排名作结论。Crossref 前八项大多不相关，不能据此声称覆盖完整。

## 可核查的新证据

| 工作 | 本次实际核查 | 与当前问题的差距 |
|---|---|---|
| [Step-Attention Refinement of DINOv3 Features for Efficient Anterior Eye Segmentation](https://arxiv.org/abs/2607.27087)，2026 | 已读取摘要与全文 HTML；使用蒸馏 DINOv3 ViT-Small，333 张私人临床图，7 类：瞳孔、虹膜、巩膜、睑结膜、泪阜、眼周、采集伪影 | 没有独立的细睫毛运动层或透明度类别；报告的 85.55% mIoU 不能证明一像素睫毛与虹膜纹理可正确分层。未核实可下载的任务权重，不进入生产替换 |
| [EyeSeg: An Uncertainty-Aware Eye Segmentation Framework for AR/VR](https://arxiv.org/abs/2507.09649)，2025 | 摘要提供不确定性建模思路；实际访问[官方仓库](https://github.com/JethroPeng/EyeSeg)，当日 main 文件树只有 README，正文仍写 code under review | 摘要“代码公开”与实际仓库不一致；不能声称已有可运行实现，更不能承诺解决彩色修图睫毛身份 |
| [Iris-SAM: Iris Segmentation Using a Foundation Model](https://arxiv.org/abs/2402.06497)，2024 | 摘要核查，针对虹膜分割微调 SAM，评估 ND-IRIS/CASIA/IIT-Delhi | 虹膜生物识别区域不是修图需要的眼球固定层与睫毛混合边；未取得本任务的权重/实片证据，仅作研究线索 |
| [Two-headed eye-segmentation approach for biometric identification](https://arxiv.org/abs/2209.15471)，2022 | 摘要明确为眼部结构、睫毛双解码器，在高分辨率近红外图评估 | 说明细睫毛需要独立材料语义，但年份较早且近红外域不等于彩色肖像。本轮没有采用其实现 |

在这次有界检索中，没有核实到可立即替换现有实现、已覆盖彩色人像细睫毛且具备可用权重的模型。这个结论不等于不存在此类研究。后续引入模型须检查具体标签和原图效果，不能只比论文年份。

## 当前实施决策

保留普通 Rust 眼部源估计、运动场与分层合成。先补**有明确语义的连续细笔划**：人确认一条睫毛的中心线和半径、可见眼球供体及需要固定的纹理区域；算法只在该源走廊内估计前景混合，不扩大到整片虹膜。未知依赖继续拒绝。

这增加的是可靠材料输入并减少标注坐标，不冒称新自动识别。相较已有离散印章，连续线必须在整段上生效、保留真睫毛运动、保持高光并通过完整眼部审查。若未降低操作成本或造成新残影，不能记作改进。后续自动化需提供相应高分辨率监督或经验证的源提示模型；不继续把 V1–V7 的颜色启发式堆叠包装为通用识别。

研究流程使用本机 `literature-review` 技能，软件方法来源：[Kassis et al., Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, 2026](https://arxiv.org/abs/2609.00065)。已由当日 arXiv 元数据核对作者与标题；该论文没有提供本修图算法的效果背书。
