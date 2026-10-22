# 三图原图目标审计：整体脸宽控制器

日期：2026-10-22。范围：已有源图、源坐标与 Rust JSON 目标生成；**无新检测、无形变求解、无照片重采样、无质量评分**。

新控制器去掉了旧近中线横向门控，因此它是一个有明确差异的目标版本。相同 `amount` 不代表相同目标请求，不能称作 same-request 数值修复。

## 输入和校准核对

使用已有 `whole-face-width-source-applicability-v2` 资料中的 woman / hair，以及其 manifest 指向的 user 冻结 A2 源资料。三张图各自读取原始 frame、128 个源轮廓点、source oval、两侧眼部 polygon。每张图的 128 点均重新与其自己的 native trace 严格相等核对；native source crop 的 RGBA 与原图对应裁剪严格相等。坐标是原照片像素索引，native texel centre 已在已有源记录中减过一次 0.5，本适配器不再转换。

校准一律是该源资料 `derive(source, frame, 100)['k']`，不使用示例文件的 0.2。两眼数据只用于重放和核对旧函数，Rust 控制器不依赖眼睛数量。

| 源图 | 显式满幅 k | bounds 横向中心 | 实际 midline_u | 方向范数平方误差 |
|---|---:|---:|---:|---:|
| user | 0.14388035339884558 | 0.755874633789 | 0 | +5.72e−9 |
| woman | 0.1308749758613116 | 7.345363616943 | 0 | −4.98e−8 |
| hair | 0.13284065085181032 | −2.875469207764 | 0 | +8.42e−8 |

`midline_u=0` 的依据是旧函数中的 `signed_radius = dot(source-origin, across)`。bounds 中心参与旧 lateral gate，但从来不是旧半径的中心，不能混用。原始 frame 保持不变，没有归一化方向，也没有校正人脸不对称。

user 源图带“AI创作”标识，本审计不将其称为真实相机照片。三图仍然远不足以证明跨场景质量。

## 确切目标差异

旧源 `whole_face_width.py` 与 manifest 中 SHA256 一致。user +100 冻结目标、woman / hair 的 −60 / 0 / +60 冻结目标均先逐项完全复现，再与 Rust 输出比较。其他档位明确记为同一冻结代码在原始源坐标上的重新计算，不冒充已有冻结数组。

下表以差异大于 `1e−9 px` 统计实质目标改变，排除跨语言运算舍入；该阈值用于区分数值噪声，不是视觉验收阈值。所有非零逐浮点差异、每点位移及局部坐标都保留在结果 JSON。

| 源图 | ±60 改变点数 | ±60 最大差异 | +100 最大差异 | 上轮廓改变索引 | 下巴改变索引 |
|---|---:|---:|---:|---|---|
| user | 20 / 128 | 1.446988 px | 2.411647 px | 117–126 | 54–63 |
| woman | 20 / 128 | 1.196120 px | 1.993534 px | 0、119–127 | 57–66 |
| hair | 11 / 128 | 0.897284 px | 1.495473 px | 0–7、126–127 | 69 |

全部实质改变都由去掉 lateral gate 解释，预测差值与实际差值的最大残差不超过 `1.14e−13 px`。旧 lateral=1 的颊侧等区间保持原有半径比例，最大数值差异同样不超过 `1.14e−13 px`。

这说明改变不是源坐标混用或 k 错配，也说明此前“只是复用旧目标”的理解不成立。新版本让近中线的顶部与下巴也服从统一半径比例，但是否更美观尚无图像结果支持。

## 上脸响应和下巴衔接

- 三图的上轮廓（归一化高度 <0.34、非零半径）分别有 64 / 59 / 57 点；−60、+60、+100 均没有意外归零点。上轮廓包括发际线和遮挡证据，不能全部称为可见额头皮肤。
- 三图零值输出都与原始源点严格相等；−60 和 +60 围绕同一源坐标的对称误差为 0。
- 下巴仍采用 `1-smoothstep(.88,1,h)`。该包络在 .88 和 1 的两端一阶导数为零；不是把相邻轮廓点各自切开后拼接。
- 在两端 `h±1e−6` 的原始 frame 探针中，跨端点位移差最大为 user `4.62e−9 px`、woman `2.36e−9 px`、hair `2.41e−9 px`；`h=1` 及其下方目标均精确不动。
- **以上只证明目标公式没有高度门限跳变，不能证明最终图像无裂缝、颈部连接正确或下巴曲线自然。** 原轮廓折点、分割误差、遮挡、求解映射和重采样都没有在这里修复或验收。

## 实际发现并修复的接口问题

1. 原来的 `1e−8` 单位 frame 容差拒绝了 woman / hair 合法的缓存 f32 frame。加入真实 frame 回归，观察失败后将容差改为 `1e−6`；仍拒绝明显非单位/非正交 frame，不归一化或改写源坐标。
2. serde_json 默认解析将 `184.30899047851562` 变为邻近 f64，导致零值 CLI 输出与原始 JSON 有约 `5.7e−14 px` 差异。真实坐标回归先失败，启用库已有的 `float_roundtrip` 后通过。该修复不被宣传为视觉质量改善。

## 图示与复现

三张图示均使用未经形变的原图；黄线为源轮廓，青线为旧目标，品红线为 Rust 目标，红圈为实际改变点。叠加结果显示差异聚集在近中线的顶部和下巴；原轮廓本身的折点仍可见。图示是控制意图审核，不是修图前后效果。

- [user 原图目标叠加](/Users/modao/work/img/test/results/semantic-deformation-20261022/controller-source-audit/user-target-overlay.png)
- [woman 原图目标叠加](/Users/modao/work/img/test/results/semantic-deformation-20261022/controller-source-audit/woman-target-overlay.png)
- [hair 原图目标叠加](/Users/modao/work/img/test/results/semantic-deformation-20261022/controller-source-audit/hair-target-overlay.png)
- [汇总及输入/代码指纹](/Users/modao/work/img/test/results/semantic-deformation-20261022/controller-source-audit/report.json)

脚本：[source_audit.py](source_audit.py)。使用已有 background-worker Python 环境，不下载依赖或模型。Rust 二进制由独立 crate 的测试构建产生。

```sh
services/background-worker/.venv/bin/python \
  experiments/semantic-deformation-20261022/rust-controls/source_audit.py \
  --output test/results/semantic-deformation-20261022/controller-source-audit
```

输出目录须不存在，以免覆盖已有审计。重跑请显式给一个新的结果目录。原始请求、CLI 输出、逐点差异和图片均只在 ignored `test/results/` 内，不进入版本库。

当前结论：源图约定及新旧目标差异已经有依据；去掉近中线门控是否值得保留，还需在明确承认目标已改变的情况下做后续效果对照，不能继承旧求解结果或旧分数。
