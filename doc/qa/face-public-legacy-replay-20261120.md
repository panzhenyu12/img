# 公共 `face.adjust` 真实回放审计（2026-11-20）

## 范围

本轮只审计普通 `face.adjust` 图（没有 `SourceLayers`、没有私有实验参数），使用仓库中已经冻结的真实检测图与原图：

- `portrait-hair`
- `portrait-glasses-street`
- `portrait-roll`
- `portrait-woman`
- `portrait-makeup`

回放命令均为当前 native `image-render`，输入为 `test/results/face-core-fresh/*-source.png`，graph 为对应 `*-faceWidth-60.json`。输出保存在本机 `/tmp/*-current.png`，照片和输出不提交。

## 观察结果

普通入口确实改变了脸部皮肤轮廓，主要五官中心保持不动；但当前不能称为可用：

| 样片 | 改变像素 | RGB 差异 >20 | 最大 RGB 差异 | 目视问题 |
|---|---:|---:|---:|---|
| portrait-hair | 3,939 | 925 | 301 | 右脸颊/耳下有高对比窄边，长发交界仍依赖估计前景/背景，强度较高时会读成旧轮廓残线 |
| portrait-glasses-street | 182 | 0 | 6 | 这一张的 `faceWidth=60` 响应很弱，眼镜和街景未被拉动；不能据此推断所有姿态通过 |
| portrait-roll | 6,574 | 647 | 76 | 滚转脸两侧有轮廓响应，但下颌与颈部的颜色过渡仍需原尺寸审图 |
| portrait-woman | 2,613 | 72 | 53 | 长发/脸颈交界有低幅度残差，普通路径没有 source-paired 的材料证据 |
| portrait-makeup | 9,719 | 988 | 93 | 右侧脸缘有连续色带，妆容/耳饰场景不能仅凭像素保护判定自然 |

`faceWidth=60` 的正向响应保留，未发现选区外写入；这些数值只说明有响应和局部性，不说明审美质量。

## 根因判断

公共入口当前在 `face.rs` 中会根据语义 mask 进入联合脸颈或旧脸部合成，但普通 graph 没有 source-paired 的真实前景/背景配对层。`face_matting::estimate` 只能从单张复合照片估计 F/B，发丝、耳下和滚转脸的遮挡关系无法由单一颜色线唯一确定，因此在新轮廓窄带会出现颜色带或旧边缘残留。

本轮分别做了两个隔离试验：

1. 将公共路径的 attachment taper 从 12% 扩到与 source-paired 相同的 16%；五张图输出无可接受改善，说明缺陷不只来自 taper 宽度。
2. 将 `repair_exposed_face_background` 强行应用到公共路径；`portrait-hair`、`portrait-roll` 和 `portrait-makeup` 的高差异像素略增，说明无可靠外部 clean donor 时，盲目射线补洞会制造新的颜色偏差。

因此本轮没有把未经真实审图验证的候选写入生产代码。

## 结论与下一步

- 普通 `face.adjust` 已证明能在真实照片上响应 `faceWidth`，但长发/耳下/妆容/滚转场景仍未达到可用水平。
- 安全的修复方向是把公共入口升级为带 source optical alpha 与 paired background 的统一材料域；仅继续调 MLS 强度、taper 或局部补洞会把残线换成色带。
- 五官、妆容和 `faceHeight/midface/doubleChin` 不能由本轮 faceWidth 回放代替验收。
