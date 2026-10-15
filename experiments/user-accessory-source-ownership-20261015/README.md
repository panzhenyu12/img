# 耳饰旁原始皮肤归属审计（2026-10-15）

只读原图 `test/results/user-face-contour-20261014/1024/canonical-source.png`，1024×1024，SHA `5f15e92b14c44a468d924af5ac985a7cc7dc1c87d8ab235f2c94b2e88b173c73`。其 AI 創作水印仍在，不能称为已验证相机实拍。采用冻结 V36 原 source physical alpha、frozen 标记和原有 v2 source support；没有重检变形结果、推理新模型、改变幅度、修图、部署或修改生产。

## 可复核结论

v2 的“432 像素”仅指 pure，另有 trusted mixed 1，总 support 为 **433**，左 264、右 169。这个集合只覆盖脸颊边缘的一部分，不能当成完整 source contour。

- 原图左右脸颊靠耳饰的亮肤色边，与脸颊内侧连续。冻结物理 alpha、旧 accessory 类和 v2 skin support 在这条边上有系统性遗漏。
- 手工只标注六条离散横向观察路径与一条内侧连接线，没有补全整个轮廓。左/右各为一个四连通集合，分别 128/107 个源像素；其中共有 **10 个 physical alpha=0 且不在 v2 support 的可见皮肤路径像素**。另外一些像素只有极低/部分 alpha。标签表示观察到皮肤成分，不是物理 alpha=1 真值。
- 每侧六个边缘探针共 12 个：11 个不在 v2 support，且全部落在既有 6px anchor distance 门槛内。11 个的 Lab 色差判据失败；左 (379,480) 的总 Lab 差 7.99 虽 ≤8，亮度差 6.55 仍被 `absDeltaL<=6` 拒绝。问题是固定内侧肤色先验漏掉明亮/低饱和度边缘，不能用 ZIM 低 alpha 直接判皮肤，也不能只靠增加 distance 修复。
- 源 support 即便已识别也不等于整片被清理：+100 的 source support 中，左 82、右 115 被原 insufficient-scene gate 挡住；-100 左 2、右 37 被挡住。清理旧位置和写新位置需要共同审核，否则残余旧边与新边并存。
- hybrid 的 81 是右侧原 parser >=.5 component 的独立 support 子集，明显小于 v2 右侧 169，更不是整条脸颊轮廓。

## 典型原图坐标

| 图像侧 | 坐标 x,y | 原图 RGB | V36 physical alpha | v2 support | old accessory | 说明 |
|---|---|---|---|---|---|---|
| 左 | 375,450 | 140,118,107 | 0 | 否 | 1 | 连续亮肤色边，原 frozen |
| 左 | 376,460 | 138,113,108 | 0 | 否 | 1 | 连续亮肤色边，原 frozen |
| 左 | 378,470 | 149,123,110 | 0 | 否 | 1 | 连续亮肤色边，原 frozen |
| 左 | 379,480 | 144,120,110 | 0 | 否 | 1 | 连续亮肤色边，原 frozen |
| 右 | 641,450 | 162,125,106 | 0 | 否 | 1 | 连续亮肤色边，原 frozen |
| 右 | 640,460 | 160,127,110 | 0 | 否 | 1 | 连续亮肤色边，原 frozen |
| 右 | 639,470 | 161,132,114 | 0 | 否 | 1 | 连续亮肤色边，原 frozen |
| 右 | 638,480 | 168,139,121 | 0 | 否 | .9971 | 连续亮肤色边，原 frozen |

完整 12 个探针的 RGB、semantic alpha、ZIM alpha、physical alpha、距离、Lab 差在 `outputs/manual-observations.json`。外侧暗过渡、贴着金属的模糊像素单独标为 uncertain；金属和耳饰孔洞里的背景示例也单独保留。未把未知区域转为皮肤。

## 文件

- `audit.py`：冻结输入诊断与计数；不产生渲染候选。
- `labels.py`：仅在原图上记录稀疏人工观察并计算既有 gate 的拒绝原因。
- `outputs/component-{1,2}-source-ownership-{1,4}x.png`：原图、physical、frozen、v2 support、ZIM。
- `outputs/component-{1,2}-semantic-{1,4}x.png`：原图与 skin/ears/accessories/hair 原分量。
- `outputs/component-{1,2}-coordinates.png`、`*-grid.png`：原始像素坐标对照。
- `outputs/component-{1,2}-sparse-labels-8x.png`：原图与稀疏观察，绿=可见皮肤路径，橙=混合/不确定，蓝=金属示例，紫=原图暗背景示例。
- `outputs/report.json`、`outputs/manual-observations.json`：原图/source-layer SHA 与完整计数。

本结果支持先修 source ownership 的诊断方向，**不是新算法已通过、不是边缘 alpha 真值、不是 8 分证据**。单张原图不能唯一分离混合像素的 alpha/前景/背景；神经补全或低重建误差均不能替代该证据。
