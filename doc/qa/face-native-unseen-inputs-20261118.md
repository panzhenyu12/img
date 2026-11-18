# 两张新源图的普通人脸接口回放（2026-11-18）

## 当前结论

固定的两张新原图均完成自动检测、源层准备和普通 rlib 的 `faceWidth=0,+30,-30,+60,-60` 回放。零值逐 RGBA 等于原图，10 张输出均成功；这不等于质量通过。614810 的正值内收在耳侧和胡茬外缘出现可见灰暗拖影，特别是 +60 的右下颌外侧，因此不能据该批宣称泛化可用。762020 尚待独立原尺寸审图，不根据小图或 changedPixels 非零判通过。

## 固定输入与策略

在任何检测或成片前记录了两张照片、SHA256、统一归一化和准备策略，文件为 `test/results/face-generalization-20261118/preregistered-inputs.json`。源图为 Pexels [614810](https://www.pexels.com/photo/614810/) 和 [762020](https://www.pexels.com/photo/762020/)，旧源引用中未发现它们。第一张近正面、有胡茬与不对称照明；第二张轻微侧转、有长发。它们不是大角度侧脸压力测试。

两张都在检测前一次 EXIF 归正、转 RGBA 并以 LANCZOS 将最长边限制为 1024，之后不再改变原图坐标。新 canonicals 分别为 1024×874、1024×682。原下载、canonical、mask、模型与输出都在 Git 忽略目录，不推送。

同一策略：自动人脸检测 → 同源多尺度后验 observed-support → 源角色与运动界面 → 附件随动许可 → observed-scale-consensus 完整头部 MEMatte → 一次完整头部 LaMa scene → 同源配对 → `prepare_face_source_replay.py` → `adjust_face_with_source_layers`。每张检测、后验、MEMatte、LaMa 各运行一次，没有按照片改提示、参数、阈值或替换样片。

## 准备过程的真实失败

保留了两个工具链故障，未将它们写成算法拒绝，也未删除后重跑模型：

1. 初次多尺度后验因历史 `.run/zim-deps` 清理后缺少 ONNX Runtime 而失败；接入已经存在的 `.run/body-source-20261117-deps` 等依赖后，复用原检测。
2. 普通 API 迁移把 `promote_interface` 实现提取到了 `promote_with_source`，旧 source helper 的源码指纹不再匹配。修正 `prepare_motion_interface.py` 同时抽取 wrapper 和实现，用已有 serde_json 依赖 rustc 编译一个小 helper，两图共用。没有跳过 hash 检查，没有 Cargo 或整体构建。

`face_pipeline.py` 新增 `--source-only`，在 pair-source 后结束，不产生历史 W10 成片。中间 native-source trace 只用于已有源估计缓存；本次所有最终图片来自普通 rlib。记录在 `source-preparation-runs.json`、`source-preparation-environment-corrected.json`、`source-preparation-resumed.json`。

## 回放与审图位置

本地根目录：`/Users/modao/work/img/test/results/face-generalization-20261118/`。

每个 `portrait-614810/`、`portrait-762020/` 下都有：

- `canonical-source.png`：冻结原图。
- `ordinary-rlib-spec.json`、`ordinary-rlib/manifest.json`：输入、五档结果、单次 native 调用耗时。
- `full-review.jpg`：source、0、+30、-30、+60、-60 全图概览，每列显示宽度 300。
- `native-plus30.png`、`native-minus30.png`、`native-plus60.png`、`native-minus60.png`：source/result 原尺寸脸颈区域，未缩放，顶部仅添加标签。
- `review-layout.json`：裁剪坐标与布局说明。

审图坐标为 source 半开矩形：614810 `[180,27,825,874)`；762020 `[317,47,775,651)`。完整输出与原图始终保留，裁剪不改变算法输入。

`execution-fingerprints.json` 记录 canonical、源缓存、公图、权限、alpha 解码误差、实际 native binary 以及每张输出 PNG/RGBA SHA256。两图的零值输出均为 0 变化像素，source hash 均保持不变。观察域和两项随动权限完整转发，没有省略消融。

下一步应由独立审图确认新增反例及错误归属；此批不调阈值、不换图，不把已有 48 图的通过边界扩展为任意照片通过。
