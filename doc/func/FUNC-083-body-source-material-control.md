# FUNC-083：身形源材料分层与同场因果对照

2026-11-19。状态：**人工源输入的单图 0/±30 有限通过；自动材料入口仍失败。** 此项不更改 UI、Go 服务或 WASM 集成，不新增“通用可用”评分。

## 目的与接口

普通自动收腰把前景袖子与另一件垂挂外套错误分给躯干后，同一形变场会错误移动它们。此项将源图的实例身份、局部遮挡次序和独立运动片区公开为可审核输入，再通过现有 `BodyLayeredSource` / `adjust_body_with_layered_source` 组合，而不更换或削弱原几何场。

源注记包含实例 ID、可见多边形、运动角色及 observed/hypothesis 的遮挡边。语义分类、人工多边形与光学 alpha 分开；同一毛衣可以具有固定弯袖和运动躯干两个片区，两片外套仍属同一 coat 实例。不可见腰侧保持未知，左侧可见衣边才进入独立响应测量。

`experiments/body-fresh-source-20261119/annotate_source.py` 冻结 source-only 注记、前层请求、后层身份与 overlay。`prepare_manual_layers.py` 顺序准备源层并输出五个 source-sized、little-endian f64 数组：`front-alpha`、`front-rgb`、`rear-alpha`、`rear-source-rgb`、`scene-rgb`。准备过程使用本地既有 MEMatte、LaMa 和 bounded source pairing，不下载模型。每个模型阶段只读源及前序源估计，不读强度、形变场或成片；隐藏支持/RGB 始终标为估计。

普通回放保留原 source graph 的 axis、region、subject/part/protect masks，只改变 `adjustments.waist` 为 `0/-30/+30`，没有以光学 alpha 替代几何 mask。源、模型、trimap、阶段输入与输出有 SHA 绑定；脚本拒绝覆盖既有阶段，人工干预及发生在自动失败之后的时序显式记录。

## 本轮证据

输入是原始 `fullbody-side-fashion.jpg` 的本次浏览器解码 PNG（1280×853）。人工源角色在出新结果前经两名 reviewer 独立复核；前层 4 px 不确定带，后层因原图粗边界偏差而在首次后层推理前另行冻结 8 px 不确定带。没有根据成片再选 alpha、场或强度。

源配对最大误差 `2.84e-14`，零值逐 RGBA 不变。两向原保护、不透明前层及零场点均未改变。自动失败中的右外套诊断区变化 `1386/1305 px`，人工方案两向为 0；前袖内部 `183/185 px` 降为 0。左可见 denim 侧缘 y430/450/470 两向分别约 `−4.065/−3.911/−2.284 px` 与 `+3.989/+3.911/+2.100 px`，保持原场在 y450/470 的实际响应。独立原尺寸 RGB 审核未见新裂缝、外套折角、袖误动或明显双边，有限接受此人工样本两向轻档。

## 尚未解决

- 自动 FASHN/Pose 派生 `partMask` 的材料归属仍错误，不能把人工对照计作自动算法通过。
- 此照片右腰被遮挡，未验证双侧真实腰宽；完整层次与隐藏纹理仍是假设/估计。
- 本轮没有强档、腿部、新独立图泛化或浏览器实时性的通过证据。
- exact 源重合成只验证数学约束；最终 RGB 的自然度必须独立验收。

详细失败与修正证据见 `doc/qa/body-fresh-source-failure-20261119.md`。源照片、数组、模型、生成图及大型测试数据不提交 Git。

## 源准备交付边界

统一的八个 CLI 阶段为 `front-prepare/front-matte/underlay/rear-prepare/rear-matte/scene/export/replay`。模型经 `model_runtime.py` 懒加载，数值阶段不再为一个 SHA 函数导入历史 ZIM/`prepare_explicit_front` 实验链。保留既有源估计和配对数学，未改模型、原冻结协议或原成片。

独立源码目录的19项定向测试通过；原前层三个阶段的13个输出文件、JSON或数组与整理后对等。冻结 v3 源层重新导出的五个 f64 文件 SHA 相同，普通 Rust `0/±30` 成片逐 RGBA 相同。该验证未重新运行模型，临时数组已清除；它证明交付整理没有改变既有结果，不增加照片泛化或自然度评分。HEAD 版本 LaMa worker 的调用兼容已验证，无需带入工作区未提交的历史 Poisson 修改。
