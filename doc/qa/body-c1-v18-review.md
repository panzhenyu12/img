# 身体织物切线连续候选 V18：未启用

2026-10-14。**候选保留为实验，不替换生产 `body.rs`，身体功能评分不变。** 合成织物上的斜线折角有明确改善，但五张实拍、十五个部位的四档回放没有稳定的整体自然度收益。不能用连续导数和合同通过代替实际画质达标。

## 问题与候选

旧分段线性逆映射在固定骨轴核心与形变区交界处有斜率跳变。把一条斜向织物线跨过该区域，会出现人为折角；即使像素位移本身连续也不能保证印花的切线连续。候选只替换横向位移延拓，保持相同目标轮廓、固定核心、部位 alpha、背景、保护区和窄带重建算法：在核心和目标轮廓之间用三次 smoothstep 延拓轮廓位移，两个接点的一阶导数与相邻区域一致。

这是确定性几何实验，没有运行 SAFlow 权重或复现其学习模型，不宣称达到论文指标。

实验回归覆盖半径 `1/5/40/200px`、强度 `-100/-60/+60/+100`、正负两侧，核对精确轮廓位置、固定核心、2000 个采样点无折叠及接点切线。旧映射先确认失败：半径40、强度-60、核心接点的两侧斜率为 `0.999999/0.8177518`；极端-100为 `0.999999/0.7291555`。候选通过这些合同及已有 body 筛选27项测试。回归仅随实验补丁保存，未向生产加入已知会失败的测试。

![合成布纹与结构背景诊断](body-c1-v18/fabric-100.png)

合成图中右列布纹的折角较平顺；中、右列新露出的背景横线仍不能正确延续，说明改逆映射不会修复窄带的结构重建。合成输入、graph、输出及 renderer 哈希在 [诊断来源](body-c1-v18/fabric-provenance.json)。

## 实拍验证

使用冻结 `test/results/body-parts-v14` 的五张实拍及十五个部位 graph，仅修改强度为 `-60/-30/+30/+60`，共60项。渲染器为独立源码快照构建的固定副本，每次子进程前后检查 SHA。完整输入/graph/renderer/body源码哈希、耗时、差分及裁切路径见 [report.json](body-c1-v18/report.json)。本轮没有重新检测，也没有以不同蒙版混比算法。完整A/B图和四档裁切保留在忽略提交的 `test/results/body-c1-v18`，`doc/qa` 只保存五张诊断图与报告。

- 所有保护像素、支持域外像素变化为0。
- 所有收缩项的原已知背景变化为0，所有项原RGBA alpha变化为0。
- 候选与基线累计245383个像素不同；这是纹理映射重新分布，不是画质提升分数。
- 逐项看了四档裁切。中低强度差异小；高强度布纹/褶皱变化更明显，但没有稳定消除旧轮廓细缝、部位语义错误及结构背景补全失败。

| 实拍/部位 | 观察与未解决问题 |
| --- | --- |
| fashion-standing / left-thigh | 裤料褶皱分布变化，植物背景窄带的细缝仍在；没有稳定自然度收益。 |
| fashion-standing / right-thigh | 裤腿真实轮廓量相近，地面边界延续未改善。 |
| fashion-standing / left-upper-arm | 袖面纹理重新分布，衣边与植物的接缝仍需更精确 alpha 和纹理恢复。 |
| fashion-standing / right-upper-arm | 衬衣褶皱的弯曲略有不同，椅背/身体接触区域仍由原部位蒙版决定。 |
| fashion-standing / waist | 胸腹布纹与褶皱有变化，收益依强度而异，不能证明更自然。 |
| fullbody-skirt / left-upper-arm | 小区域及蕾丝轮廓的细边问题没有稳定改善。 |
| fullbody-skirt / right-upper-arm | 袖与躯干同类区域归属问题仍在，换延拓函数无助于真实遮挡分层。 |
| fullbody-skirt / waist | 腰侧与衣袖遮挡、白衣与背景的旧边问题仍在；裙面内部变化不能算腰外轮廓改善。 |
| fullbody-side-fashion / right-upper-arm | 针织纹理弯曲轻微变化，袖子贴躯干的可见部位与外套分区限制仍在。 |
| fullbody-side-fashion / waist | 牛仔裤/衣边细缝及外套归属仍有痕迹；没有稳定的接缝改善。 |
| tilted-shirt / left-upper-arm | 宽袖褶皱平滑方式改变，但布料真实语义与细重边未解决。 |
| tilted-shirt / waist | 固定交叠衣袖与身体之间的分区限制没有改善。 |
| action-suit / left-upper-arm | 西装袖面较平滑，映射差异小；不足以确认真实画质提升。 |
| action-suit / right-upper-arm | 肘附近褶皱的压缩分布变化，外轮廓/背景重建没有稳定净改善。 |
| action-suit / waist | 衣襟纹理变化很弱，电脑遮挡与西装分层仍是质量限制。 |

![地面与裤腿窄带](body-c1-v18/fashion-standing-right-thigh-60.png)
![袖与腰部遮挡限制](body-c1-v18/fullbody-skirt-waist--60.png)
![侧身衣物分层与细缝](body-c1-v18/fullbody-side-fashion-waist-60.png)
![宽袖布纹](body-c1-v18/tilted-shirt-left-upper-arm-60.png)

## 效率与取舍

复杂度、65截面和逐像素采样数量保持相同；每次有效映射新增常数次乘法。交错回放的CLI总耗时中位旧561.6ms、候选535.3ms，包含启动/PNG且处于并发开发负载中，差值不作为速度改善证据。本轮未做WASM专项或GPU实验，生产代码没有启用该候选。

保留 [补丁](../../experiments/body-c1/c1-v18.patch)，以后验证更准确部位分层/窄带结构恢复时，可复查切线问题。当前先解决可见失败；不能仅凭一个更平滑的几何函数提高身体评分。

在匹配源码快照的隔离副本应用补丁，构建并冻结 renderer 后可回放：

```sh
services/background-worker/.venv/bin/python scripts/review_body_refinement.py \
  --before /absolute/path/to/frozen-baseline \
  --after /absolute/path/to/frozen-candidate \
  --label body-c1-recheck
services/background-worker/.venv/bin/python scripts/review_body_texture_join.py \
  --before /absolute/path/to/frozen-baseline \
  --after /absolute/path/to/frozen-candidate \
  --label body-c1-recheck
```

缓存的实拍输入属于本地测试产物；报告和保留的裁切不能替代源图/蒙版。重新验证应使用同一组缓存或明确报告输入变更。
