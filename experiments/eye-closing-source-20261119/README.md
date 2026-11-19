# 眼高轻闭合的源层估计实验

此目录是隔离算法实验，没有接入公开功能。原始强度固定为 −15 / −30，旧场不重算；先仅由源图准备眼睑/睫毛前景及眼球底色，再读编辑场。全部照片、层数据与输出在 ignored `test/results/eye-closing-source-20261119/`。

## 当前状态

**自动路线尚未达到通用可用验收。** 冻结 V5 已完成 makeup / beard / longhair 三张新源检查，−15/−30 经独立审图取得有限成功，但真睫毛根部的独立反例仍失败。不得将“数值重建源图”或者“没有抛异常”视为睫毛、虹膜外观正确。

人工源材料修正已实现：离散眼球/移动前景/未知标签与 alpha 分开。已知层的纹理保持和单次睫毛运动通过原定门槛；用户照片的眼球高光修正、makeup 的一根细睫毛轻档移动也经独立审图接受。人工身份确认不记作自动识别成功，颜色模型不确定仍报告。范围见 [FUNC-078](../../doc/func/FUNC-078-eye-source-material-correction.md) 与 `MANUAL-MATERIALS.md`。

独立 `test_observed_ocular.py` 不属于估计器实现：它从已知层生成带竖向纹理的眼球及斜向抗锯齿睫毛，使用独立双线性 oracle 检查最终 RGB 和暗线质心。两项测试的门槛保持不变。

| 版本 | 变化 | 可见亮/暗虹膜纹理 | 真睫毛最终 RGB |
|---|---|---|---|
| V1 `source_estimator.py` | 同法向眼内供体 + 三色混合 | 失败：高光变为移动前景 | 常量 B 用例的误差抵消不能证明层正确 |
| V2 `observed_ocular.py` | 观测虹膜锚点 + 与眼睑相连的黑帽分量排除 | 失败：虹膜纹理被整分量误排除 | RMSE 4.936，但质心误差 .264 px 超过 .25 px |
| V3 `observed_ocular_v3.py` | 要求眼裂外可见深色睫毛根 | 通过 | 失败：合法睫毛可以没有眼裂外可见深色根 |
| V4 `observed_ocular_v4.py` | Hessian 细暗线、两侧反差和向眼内线共识 | 通过 | RMSE 9.431640，未过 5.650029 |
| V5 `observed_ocular_v5.py` | 保持 V4 准入，固定 1 px 前景采样支撑；排除污染供体并从细线两侧估计 B | 通过 | RMSE 5.899756，未过 5.650029 |
| V6 `observed_ocular_v6.py` | 源局部仿射面提供单侧底色估计 | 失败：释放假暗线支撑后高光误差 170.29 | RMSE 1.636500，质心误差 .0612 px，通过；整体仍拒绝 |
| V7 `observed_ocular_v7.py` | 固定稳健表面估计，降低亮供体污染 | 失败：高光误差 142.65，颜色拟合不能可靠判别材料身份 | RMSE 1.877722，质心误差 .0364 px，通过；整体仍拒绝 |

V5 明确承认隐睫毛下眼球是估计值，不是观测光学真值。它没有使用编辑后输出、请求强度、测试真层或真遮罩构造源层。V4/V5 失败版本留存，不改测试迁就算法。

## 已坐实的原因

1. 原同法向三个供体会落在睫毛上；独立源 u25 的供体 RGB 为约 `[23,27,34]`、`[34,42,53]`、`[57,72,90]`，真实眼球约 `[70,88,110]`。median 不能去掉三者共同的污染。
2. 暗线 ridge 仅表示中心，不包含所有抗锯齿混色边；直接将其余区域升级为 `E=1` 会冻结部分睫毛。
3. V5 修正前两点，但在靠近眼睑的根部，两侧干净眼球供体并不同时存在。例 u22,v−6 的真实 lash coverage 为 .594，双侧供体无效时旧核心仍给 E=1。剩余失败不是性能或打分问题。
4. 连续眼睑闭合不等于不会露出旧睫毛下的眼球。源图内细睫毛移动产生的微小显露允许受限估计；源眼睑外完全隐藏的眼球不能冒称已经观测。

## 冻结版本

- V5 SHA256: `b50948136899162035269abfa24a5ea78aeffaa7004219280935bdb466367da9`
- V4 依赖 SHA256: `6888e9d65b316443c35dee02484e0aca2b5a69b6b154b122a1ea3558eea0186e`
- V1 依赖 SHA256: `b9b78689f392bf95d60af1ce88fa5c41c2c72b840a2d1c958ff01c62ff58239b`

调用接口：`observed_ocular_v5:estimate_observed_ocular`。`run_closure.py` 在读取场之前只调用一次源估计器；同一源层用于 0 / −15 / −30。

冻结协议：`test/results/eye-closing-source-20261119/fixed-new-source-protocol.json`。独立 V5 图和失败记录：`test/results/eye-closing-source-20261119/independent-observed-ocular-v5/`。

旧 user/dark 场的实际中点眼裂高度变化已测量：−15 为约 3.4%，−30 为约 6.8%。冻结场双线性采样的眼角位移约 .006–.018 px；不能声称位级固定或完全闭眼。完整量测在 `fixed-field-midpoint-motion.json`。

## 显式复现自动路径反例

默认仍为 V2（`observed_ocular:estimate_observed_ocular`）。用环境变量选中冻结版本，不修改独立 fixture、断言或门槛：

```sh
EYE_SOURCE_ESTIMATOR=observed_ocular_v5:estimate_observed_ocular services/background-worker/.venv/bin/python experiments/eye-closing-source-20261119/test_observed_ocular.py -v
EYE_SOURCE_ESTIMATOR=observed_ocular_v6:estimate_observed_ocular services/background-worker/.venv/bin/python experiments/eye-closing-source-20261119/test_observed_ocular.py -v
```

两条命令目前都应报告相应自动算法失败，原因分别见上表和 `V6-NOTE.md`。人工材料修正使用独立 `test_source_material.py`，不能用人工标注的结果替换或注销自动路径反例。
