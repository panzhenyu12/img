# FUNC-034 上额宽度与太阳穴共享材质形变

## 功能目的

让 `foreheadWidth` 和 `temples` 在发际线位于皮肤/头发交界的照片上真正产生轮廓变化，同时保持外层头发、背景、五官、耳饰和用户保护区域不动。

## 默认算法

`face.adjust` 在检测到独立的 `face`、`faceContourMask`、`hair` 栅格时，先在头部不透明区域内寻找与 `faceContour` 连通的窄发际带。候选像素必须同时满足：

- 头部 alpha 不透明；
- 向外有足够的头发证据；
- 向内有皮肤轮廓证据；
- 与观测脸部轮廓的距离在有限 band 内；
- 不属于眼、眉、鼻、唇、牙、耳饰、耳朵或 `protectMask`。

满足条件的像素只改变语义 ownership：加入 `skin`，从 `hair` 移除。源 RGB 不被预先绘制，随后仍由既有联合脸部的正值速度场、固定材质约束和 source-forward 采样完成输运。递归调用用 `__upperSharedPrepared` 防止重复 promotion；`__legacyUpperShared` 仅供历史回放跳过该逻辑，不能作为公共参数。

## 方向约定

- `foreheadWidth > 0`：上额向外变宽；`< 0`：收窄。
- `temples > 0`：太阳穴向内收；`< 0`：向外扩。
- 该 promotion 也适用于与脸宽、下颌等轮廓参数的组合请求。递归保护保证同一发际带只释放一次，所有参数随后进入同一个联合 source-forward 场，避免上额控制在组合请求中被固定头发吞掉。

## 回归与验收

- `face_upper_width` 合成测试：双向方向、外部逐字节冻结、局部 Jacobian 正值、promotion 幂等。
- `face_hairline_regression::upper_width_and_temples_use_the_public_source_forward_path`：公共入口不再 0 像素响应，并检查眼、眉、鼻、唇、耳饰保护区域逐像素不变。
- 四张缓存人像（dark、woman、hair、makeup）各跑 `foreheadWidth`/`temples` 的 `±100`，16 次默认路径均完成；眼睛、眉、鼻、唇、牙区域变化为 0，头部 alpha 外变化为 0。
- 四张缓存人像的 `faceWidth=60` 与 `foreheadWidth=60`/`temples=60` 组合回放显示上部变化重新出现，选区外变化为 0，五官语义多边形内部变化为 0；详见 [组合实图回归](../qa/upper-face-combined-real-audit-20261027.md)。

当前状态是“执行和保护合同通过，视觉自然度继续验收”。尚未把该功能评为 8 分；必须继续补 `±30/±60`、侧脸、眼镜、耳饰、长发遮挡和原尺寸边缘连续性复核。

## 相关代码

- `core/image-core/src/face.rs`
- `core/image-core/src/face_upper_width.rs`
- `core/image-core/src/face_hairline_regression.rs`
- `doc/qa/upper-face-shared-candidate-20261027.md`
