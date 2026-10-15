# 局部解剖边界约束实验 B：拒绝接入

结论：在用户原图的冻结 V36 source alpha、foreground 与 background 上，仅修正耳根/耳垂附近目标轮廓，不能消除现有黑楔、细皮肤条与双边。12 个消融均不满足视觉要求，不接入生产。

## 假设与实验控制

假设 B 是“连续位移场的目标轮廓仍违反耳垂连接，固定连接节点并建立 C2 位移过渡，可以保留瘦脸量同时改善耳边形状”。

用原图耳朵语义与实测皮肤轮廓的接近关系自动提取接触节点，未使用本照片坐标特判。左侧接触行 290–415、右侧 299–427；这些是本次运行的输出，不是预设。语义节点尚未具备人工解剖真值，置信度只表示“观察到语义接触”。

候选在耳垂行将水平位移与一、二阶导数夹紧为零，用五次平滑函数在脸高的 0.08、0.12、0.16 范围内接回 V36 目标。空间衔接使用紧支撑 C2 水平后变形，再与冻结 V36 inverse 复合。没有改 source alpha、背景、前景颜色、检测图或生产代码。

- 原图：`test/results/user-face-contour-20261014/1024/canonical-source.png`
- source SHA-256：`5f15e92b14c44a468d924af5ac985a7cc7dc1c87d8ab235f2c94b2e88b173c73`
- 图、source layers、inverse、baseline 都来自冻结 V36；±30 由同一诊断二进制导出，并核对与 V36 baseline RGBA 逐字节一致。
- `run.py` 是探索实现；`verify.py` 再从实际 inverse 求解 forward source probes，核对实际位移而非仅核对请求目标。

## 结果

下表使用最长衔接消融 0.16（并非入选方案）。黑楔指标为固定耳旁区域内“source 亮度 > 65、输出亮度 < 50”的新暗像素，属于失败代理量，不是语义黑楔真值。

| 参数 | 实际位移保留比例（两侧较低值） | V36 新暗像素 | 候选新暗像素 | 新增软饰品像素变化 |
|---|---:|---:|---:|---:|
| +30 | 87.50% | 669 | 610 | 148 |
| -30 | 88.35% | 0 | 0 | 214 |
| +100 | 91.20% | 1103 | 1037 | 136 |
| -100 | 91.01% | 0 | 0 | 219 |

所有 12 个消融：

- 实测位移保留 87.50%–97.78%，并非靠功能近似关闭掩盖失败。
- source alpha / foreground / background 的 SHA-256 前后相同；输出 RGBA alpha 与原图一致。
- 完整 inverse 内部 Jacobian 翻折数为 0；独立原像素 opaque metal 核心没有变化，耳核心也没有新增变化。
- 软饰品语义区新增变化为 101–219 像素，仍不能作为安全候选。
- 加入夹紧后，多数方向相对 source 的局部新增曲率 RMS 更大。连接约束把位移集中到较短过渡段，未产生更自然的脸侧曲线。

1×全脸与 2×耳旁复核均可见失败：+100 的原位细皮肤条仍与内移皮肤形成双边，黑楔持续；-100 耳根仍有锯齿状断续，候选没有得到明显改善。扩脸的新暗像素代理量为零，并不代表扩脸视觉合格。

## 证据与限制

- 实验入口：`experiments/face-attachment-boundary-v1-20261015/run.py`
- 独立实测：`experiments/face-attachment-boundary-v1-20261015/verify.py`
- 全量指标：`experiments/face-attachment-boundary-v1-20261015/report.json`
- 实测核验：`experiments/face-attachment-boundary-v1-20261015/verification.json`
- +100 左耳：`experiments/face-attachment-boundary-v1-20261015/clamp-0.16/100/left-2x.png`
- -100 右耳：`experiments/face-attachment-boundary-v1-20261015/clamp-0.16/-100/right-2x.png`
- 全脸：每个 `clamp-*/<amount>/face-1x.png`

这不是对所有解剖约束方案的否定，也没有证明每个黑像素都由同一原因造成。它证伪的是“冻结当前源图层，仅通过耳垂曲线夹紧即可解决当前可见症状”的具体假设。后续优先复核 source support 中遗留的皮肤细条、混合饰品边缘及真实可见性；不继续凭曲率指标反复微调此分支。
