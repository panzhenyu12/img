# 联合 face + upper-neck 物理前景 v19（隔离，一次假设，未验收）

结论：联合运输明显减少旧 v17 的大黑楔、斜向脸/脖子材质错接，保留约20px脸部请求幅度；仍有12px原皮肤岛、耳根细黑点和边缘阶梯，不能评到8分，未合入生产或公共WASM。独立原尺寸审查见 [root QA](../face-joint-v19-root-review/review.md)。本次只验证同一用户源图与同一缓存detector ±30/±100四档，不增加六档、九参数或改阈值重试。源图含AI创作水印，不称为已验证相机实拍，裁切不增加独立样本。

## 改变与证据边界

复用v17 face目标、Clarabel侧轮廓可行QP、Wendland RBF/SVF和同一f32逆流。`max(faceContourMask, semanticMasks.skin)` 作为有限启发seed：skin是parser原始class组1/10/14聚合后阈值化，保留当前face-skin/neck交界，但两个已阈值化group取max不是通用raw概率union或物理真值。要求neck和skin raster证据，从目标源face种子寻找可信连通skin，BFS不穿显式ear/hair/accessory/full protect；未连通的高alpha皮肤岛排除，只留邻接软trimap，再用原RGB `refine_portrait_alpha`。没有绘制neck envelope或新增分类概率/necklace mask。

可信neck处允许通过原head alpha=0裁切；上颈和face纹理/标签随同一flow，运动向下逐渐归零。无v18硬neck pin；joint停用旧neck独立relighting，私有v17已不使用第二个jaw alpha clip。物理alpha覆盖可见连接，运动falloff不是人为透明neck截止线。原region/mask/protect完全不变。CLI原协议不允许新的实验参数，初次验证因此被拒绝而没有生成图；之后只用隔离native环境 `FACE_JOINT_NECK_EXPERIMENT=1` 启用该分支。公共协议/生产代码/WASM均未扩展。

## 实际边缘与保护

| faceWidth | 源face polygon实际最大位移px | 原scene RGB改变 | 数值minJac |
|---:|---:|---:|---:|
| +100 | 20.075 | 0 | 0.362 |
| -100 | 19.838 | 1474 | 0.620 |
| +30 | 6.030 | 0 | 0.693 |
| -30 | 6.016 | 358 | 0.873 |

此表自动由 [四档report](report.json) 的冻结输出生成；精确数值以report为准。face polygon实际移动是source标签几何，不能替代最终皮肤外轮廓。

联合物理alpha的真实外边界另测 `.5` crossings：在y545，+100源联合span `[408,613]` → `[429,591]`，face语义span `[412,599]` → `[432,580]`；y560联合 `[407,614]` → `[429,594]`，face语义 `[428,585]` → `[446,568]`。两者分开记录，不能把face/neck内部标签边缘误称为外皮肤轮廓。y620联合 `[391,635]` → `[392,635]`，上颈运动自然减小；y700之后源RGBA改变0。原face中心y540与neck中心y620在同一最大物理组件（源141172px，+100后138125px）；物理估计仍有多个<=5px小岛，不能宣称整个mask严格单组件。完整逐行union/label量化：[边界报告](joint-boundary-report.json)。alpha仅局部RGB估计，不是独立physical gold；crop底723，源parserneck延伸910，运动已在crop/解析截断前实际归零。

四档ear/hair/accessory source RGB改变0、9个实际五官内部0、RGBA alpha0，原region外0。收缩原scene改变0，扩张原scene新增覆盖单独计量，不能同时承诺全原scene exact。原物理alpha=0/new>0且原person/head=0的候选覆盖：−1001501px，−30390px；真实RGB改变1474/358（阈值与输出舍入不同）。没有移走远背景。F/B估计本身未接donor_frozen/person/ROI，所以输出固定exact不等价于全供体权限证明；这一点仍未完整验收。

项链：source-only低彩高亮花瓣启发mask，共5358px，原照x323..400及635..715、y645..720侧弧；overlay已目视确实覆盖首饰上缘，四档这些像素RGBA变化0。独立root也查看了整首饰带/diff，没有发现移动。当前accessories不含独立necklace15，启发mask不是全物件gold，不能推广遮挡项链。见 [源mask](necklace-source-heuristic-overlay.png)、[像素记录](necklace-heuristic-report.json) 与 [root首饰对照](../face-joint-v19-root-review/review.md)。

## 残留点与ROI

+100/+30左侧残留原图`x407..408,y547..552`12px。所有head/skin/face/neck/hair/ear/accessory=0；source/new物理alpha、joint seed/owned均0；person约.239–.724<.8，contextFrozen=false，输出RGB与源完全相同。它是源观察未被解析/物理edge支持纳入擦除域，**不是**scene补全生成、explicit固定或fold。不扩大support去抓点；[support诊断](residual-island-support-report.json)、[root坐标](../face-joint-v19-root-review/residual-dot-coverage.json)。耳根细点和硬边也未解决。

理论falloff中心start≈614.77/end≈722.39，而原ROI底717.814，不能按代码注释声称end先于ROI。debug-only新增同一真实f32 grid/正反RK4 mapper探针，四边各±2/±1/±.5/0px实际velocity和mapping位移全部精确0；原因是本图compact RBF支持已归零。此图未发现ROI断层，不代表其他裁切有连续性保证。仪器版本四档PNG与原冻结候选逐字节相同：[输出对账](roi-report.json)、[边界映射报告](roi-continuity-report.json)、`roi-debug-only.patch`。

## 测试与快照

env开关后的功能源码（`face-tested-before-roi-debug.rs`）完整face单测 **40 passed / 0 failed / 1 ignored**。新增joint测试直接用内部params入口，验证必须已有neck/skin raster、face-neck内部无scene洞、低颈源像素exact和脸缘真收缩；CLI四档单独走native env路径。旧静止neck函数测试保留，不删除断言。`final-env-snapshot-unit-test.log` 对应这个功能快照，不能套给后来仅增加ROI trace的源码声称重跑过；trace单独patch与四档PNG parity证明无图像变化。`src/` 保存含trace最终可重建crate，另保留测试前trace的完整face源码和hash。预env单测日志只作为历史记录。

数值四档112320stride2 probes无fold，最大正反误差<=.000367px。CLI总3.00–4.72秒含PNG/graph/调试导出，机器同时运行测试，不当稳态或WASM帧率。没有本次多原片、实时性能或WASM parity验收。

![+100原图/v17/joint因果](faceWidth100-jaw-neck-causal-2x.png)

![−100原图/v17/joint因果](faceWidth-100-jaw-neck-causal-2x.png)

完整源码/依赖、各快照patch、input/binary/four outputs/debug/test SHA见 `provenance.json`；复现脚本与限制：[隔离crate](../../../experiments/face-joint-v19-20261014/README.md)。本轮到此冻结，只有一次联合假设候选，不因失败持续调阈值。

记录纠错：主QA初稿手填 −100/+30 三项数值抄错，已改成脚本从report自动生成。冻结候选未改写：四档文件SHA同时匹配首次render report与独立trace输出；[最终完整性](output-integrity-final.json)。此纠错不改变原尺寸图像与独立目视。
