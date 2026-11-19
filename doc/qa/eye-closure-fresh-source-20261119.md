# 三张新源照片的轻闭眼实验验收

2026-11-19。冻结 V5 源材料估计器，在 makeup、beard、longhair 三张未参与本估计器调参的照片上，由新检测观测自动准备源层，再回放 `eyeHeight = 0 / −15 / −30`。**三张指定照片的 −15/−30 轻闭眼取得有限视觉成功；算法尚未达到通用可用，公共 `feature.adjust` 没有开放本实验。不能据此宣称整体达到 8 分。**

独立 reviewer `/root/global_review_1119` 查看了三张 −30 全脸和六只眼睛在 −15/−30 的共十二张局部三列对照。眼裂轻收有实际可见响应，虹膜与反光保持，未见新增硬缝、双睫毛或脏补块。独立已知真值睫毛反例仍然失败，不因这三张照片外观通过而撤销。

## 输入身份与冻结规则

| 独立源 | 原图 | 尺寸 | 源 SHA-256 |
|---|---|---|---|
| makeup | `test/images/portrait-makeup.jpg` | 1280×1128 | `1b2a0147296ce870f74cbace5bfa2e2d147a8c0b384d79794bc4423266f3fd70` |
| beard | `test/images/holdout-20261118/portrait-614810.jpg` | 1280×1093 | `c9be74846fbe24d408b362b1d9fe50daa5e0e73745e9b2b32a4b57ae027892b4` |
| longhair | `test/images/holdout-20261118/portrait-762020.jpg` | 1280×853 | `b803b2c9e5c0f7c3e29aef81d9d7d82274d3fb4c04ea1f28ad8d82efba06df1a` |

观测分别位于 `test/results/eye-closing-source-20261119/new-observations-{makeup,beard,longhair}/observations.json`。它们各含一次完整原图 MediaPipe 观测，三个观测 SHA-256 依次为：

- makeup：`84eabfe4651fb05b4868914ac13e8917b5e3a9b55e0e4128b1f6043967199545`
- beard：`c4cfa0d94bdc63ed94f68c6abd2bfd52ae72c7d05b4e178ab9dafae0c60c7e53`
- longhair：`34d7da7fd6c54993b712bb8649f3f709d39b2272fdd55834321f628478c127a2`

图源、角色、强度和失败保留规则在候选出图前写入 `test/results/eye-closing-source-20261119/fixed-new-source-protocol.json`。makeup 的旋转派生图 roll 不算独立样本；戴眼镜的 man 不在本轮材料合同内，两者没有进入此次三图验收。

每张图只有原图 RGB 和该图的新观测进入源准备。每眼估计器调用一次，两眼合计两次；之后三个强度共用同一源层。不得读取旧 user/dark 的 state、graph 或 field，不得按照图名分支，不得看编辑结果后调源估计、阈值或强度。运行前后检查输入、执行文件和所有实际 Python 算法依赖的 SHA-256 一致。

固定算法入口为 `observed_ocular_v5:estimate_observed_ocular`：

| 代码 | SHA-256 |
|---|---|
| `observed_ocular_v5.py` | `b50948136899162035269abfa24a5ea78aeffaa7004219280935bdb466367da9` |
| V4 依赖 `observed_ocular_v4.py` | `6888e9d65b316443c35dee02484e0aca2b5a69b6b154b122a1ea3558eea0186e` |
| `source_estimator.py` | `b9b78689f392bf95d60af1ce88fa5c41c2c72b840a2d1c958ff01c62ff58239b` |

V5 保持 V4 的细暗线判定，给已承认的细线增加固定 1 px 采样支撑，并排除前景污染后从细线两侧估计眼球底色。它承认被睫毛遮住的底色是估计值，不宣称从单张 RGB 获得光学真值。

## 从新观测到结果

入口为 `experiments/eye-closing-source-20261119/run_closure.py`，源几何为同目录 `source_geometry.py`。

1. 检查原图尺寸、完整观测、原图与观测绑定 SHA-256。多脸必须显式选择，不猜人物。本入口只接受不透明 RGB；EXIF 旋转或部分透明源拒绝，不能无声转向或丢 alpha。
2. 复用 `source_curves.frame/fit_curve` 的纯计算函数，根据本次原图定位眼睑曲线。曲线是源几何估计，不直接等同材料 alpha；不调用旧样片准备函数。
3. 在读取任何目标 field 前，分别准备左右眼的曝光权重、固定眼球底色及其支持；双眼支持重叠时拒绝，不最后写入覆盖。`prepare_sampling_support` 为双线性足迹提供显式源支持。
4. 用本次 eye/brow 的规范有序周界与 face oval 生成三个新 graph，不用凸包替代实际周界。调用已有 native canthi field runner，不重新编译。
5. 使用同一份源层和各档新 inverse，复用 `layers.compose` 合成：眼睑残差与曝光权重移动，眼球底色保持在目的位置。未知材料、无界 RGB、非法场均拒绝，不降低强度掩盖失败。

native 可执行文件为本地现有 `test/results/eye-height-layers-20261117/canthi-field-candidate/eye-field-tests`，SHA-256：`6d2a7c4438ac49d0493614768927ff4587cd7f864fc8a0c7999eaa6dfffc57ce`。它是冻结实验场生成器，不能当成当前生产功能已支持眼高调整的证据。

## 解码与坐标域

所有源层和最终候选合成都使用同一份 Pillow RGB 解码。native 单平面对照由 native 自己解码同一 JPEG；三个输入的两解码器最大 RGB 差均为 4，因此不能要求二者逐像素完全一致来判断零操作。native 的零值 RGBA 恒等由其原生测试断言；Python 合成零值在自己的解码域单独确认逐 RGB 恒等。native 单平面对照明确标注解码域，不将这类解码差当成算法改善。

源几何定义 `P = MediaPipe normalized × 原图尺寸` 为物理坐标，数组索引 `(x,y)` 对应物理中心 `(x+.5,y+.5)`；源采样函数会减 `.5` 转为索引。冻结 native 场却在整数 pixel-index 上求值，并直接将 graph normalized 乘尺寸作为特征点。

因此仅在新 graph 的特征点处转换一次：`native normalized = (P − [.5,.5]) / 原图尺寸`。原观测、源曲线和导出的 inverse 不再平移，避免双重转换。转换后越出像素中心范围的点拒绝，不 clamp。外围 region 仍采用原始 normalized FACE_OVAL bounds 加 20% padding 的准入边界合同，不套用特征点转换。这次适配没有改旧 user/dark 的历史场。

未提供 hair/ears/accessories 固定遮罩时，冻结 native 使用空 stationary 向量，查询缺省为 false。runner 将这一明确合同转换为全 false；任意非零但不足整幅的 raster 仍拒绝。该适配不是增加保护像素或修改 field。

## 数值结果与独立视觉范围

三个零值候选均与各自 Python 源 RGB 完全相同；六个非零场的 `foldPixels=0`，最大反演误差不超过 `0.0000683 px`，无未知眼球材料采样。仍完全可见的**估计**眼球核心 RGB 变化为零；这项数值取决于估计的材料归属，不能替代真实睫毛审查。

| 来源 | 独立视觉观察 | 通过范围与限制 |
|---|---|---|
| makeup | 长睫毛、妆线连续；虹膜/高光保持；眼裂轻收，未见新增硬缝、双线或脏补块 | 仅该源 −15/−30 轻闭眼；不证明任意长睫毛根部均正确 |
| beard | 蓝色虹膜与大反光保持，眼裂有克制的闭合响应，未见新增断接 | 仅该源 −15/−30；不等于眼球所有被遮纹理已恢复 |
| longhair | 重眼线与轮廓连续，轻闭响应可见，没有明显新补块 | 源图细睫毛模糊，不能充分验收细睫毛根部重建 |

此结论不外推到完整闭眼、睁眼、强档、眼镜、任意遮挡或其他五官操作。−15/−30 是固定请求值，不代表眼裂高度缩小 15%/30%，也没有宣称完全闭眼。

## 仍失败的独立反例

`test_observed_ocular.py` 从独立已知源层生成沿闭合方向变化的眼球纹理和斜向抗锯齿睫毛，使用独立四邻双线性 oracle 比较最终 RGB 与暗线质心。生成器、强度、判定窗口和门槛没有迁就 V5 改动。

V5 通过仍可见虹膜的亮/暗纹理保留测试，但真睫毛最终 RGB RMSE 为 **5.899756**，仍超过预先固定的 **5.650029** 门槛。根部附近不总能找到细线两侧干净眼球供体，部分真实睫毛还会被错误当成完全可见眼球。三张照片有限视觉成功不能注销这个已知失败。因此源层估计还需继续修正，本轮不开放公共功能、不评整体 8 分。

`test_run_closure.py` 的 **9 项通过**只验证输入/回放合同：多脸选择、尺寸与绑定、物理坐标、native 单次转换与越界拒绝、region 不移位、固定闭合强度、几何诊断、空 stationary 向量等。这些是防止错误归因的检查，不是九个美颜功能达到效果标准。

## 证据与复现

最终验收目录为：

- `test/results/eye-closing-source-20261119/fresh-v5/makeup-r3/`
- `test/results/eye-closing-source-20261119/fresh-v5/beard/`
- `test/results/eye-closing-source-20261119/fresh-v5/longhair/`

各目录包含全图、同范围眼部裁图、新 graph、native field、一次源准备以及完整哈希报告。汇总为 `fresh-v5/summary.json`。初次 `makeup/` 被 runner 错误的非空 stationary 假设拒绝；`makeup-r2/` 是外围 region 合同修正前的中间结果。它们都保留原因，但不参与本次验收；没有据其输出修改 V5 或阈值。

从仓库根目录运行，输出目录必须不存在：

```sh
services/background-worker/.venv/bin/python experiments/eye-closing-source-20261119/run_closure.py \
  --source test/images/portrait-makeup.jpg \
  --observations test/results/eye-closing-source-20261119/new-observations-makeup/observations.json \
  --output /tmp/eye-closure-makeup-new-replay \
  --field-runner test/results/eye-height-layers-20261117/canthi-field-candidate/eye-field-tests \
  --estimator observed_ocular_v5:estimate_observed_ocular

services/background-worker/.venv/bin/python -m unittest discover \
  -s experiments/eye-closing-source-20261119 -p test_run_closure.py -v
```

`--field-runner` 是显式本地外部输入，不在代码提交中携带二进制。照片、观测、层数组、输出图及 native field 留在忽略目录；提交本小步 Python 代码也不意味着全新 checkout 已有这些本地验收资产。运行说明另见 `experiments/eye-closing-source-20261119/RUNNER.md`。
