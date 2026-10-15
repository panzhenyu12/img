# 有颈部支持的目标下颌 opacity 约束 v11

日期：2026-10-14。已进入 `face.rs` 的生产实现，但只修复一个可重现的边界问题；没有把脸型或下巴长度记为 8 分。

## 实际改变

饱和的面部解析 alpha 在下颌的 face/neck 混合像素上可能留下不透明小凸起。直接反向采样会把这些凸起平移到新下巴下缘，因此目标曲线虽然移动了，输出仍有一排皮肤色齿。

本轮不改目标曲线、MLS 几何强度、源物理 alpha、F/B 估计或颈部光照。只在目标下颌附近给采样 opacity 增加几何上界，随后按同一比例缩放预乘 RGB/RGBA，使纯前景颜色保持原值。新露底层仍交给既有 neck/hair/scene 分层重建。

条件全部成立才执行：已有面部 raster；像素在解剖高度的下 24%；距目标曲线不到 8px；对应原曲线沿外法线的内收超过 1px；原边界外侧 6px 有完整的已见 clean neck 支持（双线性 coverage >= .999）。原检测、五官、用户冻结、ROI 和配饰保护仍优先。手工/旧 polygon 没有这一额外约束；缺少可靠颈部支持时保留原路径。扩大下巴和脸宽的方向没有启用这一上界。

## 原尺寸效果

实际核对 makeup、woman、hair 的 -60/-100，以及六图强档 faceWidth/jaw/chin 对照。makeup 和 woman 的强缩短新边缘比 baseline 干净，原齿状皮肤凸起明显减少。root 又独立看了 makeup 4x 与 woman 原尺寸图，认可这一窄范围净收益。

![makeup 原尺寸](portrait-makeup-chin--100-comparison.png)

![makeup 下缘 4x nearest](portrait-makeup-chin--100-edge-4x.png)

![woman 原尺寸](portrait-woman-chin--100-comparison.png)

它没有生成被原下巴遮挡的真实皮肤纹理或三维阴影。旧下颌暗弧、颈部色泥仍在；hair 耳下/发丝交界的黑三角仍未解决。当前不能宣称缩下巴已达 8 分，脸宽/下颌宽度也没有因本次小幅边界改动获全面加分。

![hair 原尺寸，仍有旧颈部斑纹](portrait-hair-chin--100-comparison.png)

## 回归与保护

新增两个有明确真值的 Rust 回归：

1. 支持 neck 的收缩下颌，在几何目标下缘外即使源采样 opacity=.9，也必须降到目标 alpha；同时未混色的前景 RGB 必须保持 [220,160,100]，内部样本保持 exact。
2. 全无 neck 支持、支持只有 .998、脸颊上缘、下巴扩展均必须逐值保持原样。

隔离 target 的面部回归 **28 passed，1 ignored**（手工性能 benchmark）；`cargo clippy --all-targets -- -D warnings` 通过。生产源码已冻结供主任务做全量 native/WASM 检查；这里只报告本分支实际执行的检查。

六张输入的 faceWidth/jaw/chin × ±60/±100 共 **72/72** 保护检查通过；追加 chinWidth/chinPoint 同四档的 **48/48** 也通过，共 **120/120**。保护条件没有缩小：ROI 外、实际五官 polygon 内、hair/ears/accessories、用户完全冻结区均逐像素恒等；收缩时 scene/head-neck 外部也遵守原合同，允许的既有语义 neck-lighting 范围未扩大。portrait-roll 是 makeup 派生图，六张输入并非六个独立人物。

这不是 120 项自然度验收。本轮只对主要强档和三图缩短做了目视评审，未给未看子项独立分数。72 项中 34 项有像素变化，全部在收缩方向；扩展方向 candidate 与 baseline 完全相同。变化位置与数量见 [pixel-deltas.json](pixel-deltas.json)，不能把数量当成画质提升指标。

报告：[72 项](final/replay-report.json)、[追加 48 项](additional-controls/replay-report.json)。完整源图、输出和只读二进制副本保留在忽略目录 `test/results/face-edge-v11`；文档只保存代表图，避免重复存放全套图片。

## 性能与 provenance

- baseline CLI：`.run/face-edge-v11-baseline-image-render`，SHA-256 `4d856424504dd83a7c1c3bf1a72b10608a04045031ece71fffa4cfe98edb8941`
- final CLI：`.run/face-edge-v11-final-image-render`，SHA-256 `c547b29de009b01dd08d92598970b6c0c171bebfb8ae13722b599035578ff4e5`
- CLI 72 项中位数：baseline **411.135ms**，candidate **414.985ms**，包括图像/JSON 解码、校验、编辑、PNG 输出。
- 同进程 native 热运行三图 chin=-100：makeup **487.9 → 471.9ms**、woman **377.9 → 409.3ms**、hair **400.4 → 391.8ms**。每图首轮后重复五次，中位数见 [hot-timings.json](hot-timings.json)；此范围含 graph parse/校验/编辑/合成，不含识别、文件 I/O。

机器有并发任务，这些数据不证明稳定加速，也不证明浏览器实时性达标。源文件 SHA 和 accepted patch 见 [source-provenance.json](source-provenance.json)、[accepted-jaw-opacity.patch](accepted-jaw-opacity.patch)。回放期间每次执行前后都验证只读 binary SHA，构建目录无法覆盖评测 executable。

复现 72 项（依赖已有 fresh mask 缓存）：

```sh
services/background-worker/.venv/bin/python scripts/review_face_refinement.py \
  --baseline .run/face-edge-v11-baseline-image-render \
  --candidate .run/face-edge-v11-final-image-render \
  --baseline-label 27f43bf --allow-neck-lighting \
  --keys faceWidth jaw chin --values -100 -60 60 100 \
  --output test/results/face-edge-v11/reproduce \
  --doc-output doc/qa/face-edge-v11/reproduce
```

## 三个拒绝的候选

| 候选 | 检查结果 | 拒绝原因与保留证据 |
|---|---|---|
| alpha 沿切线均值 | 6/6 先行保护通过 | 齿状边只变模糊，hair 黑三角不改善。[patch](candidate1/face-tangent-alpha.patch)、[报告](candidate1/replay-report.json) |
| 原颈部精确 q 的低频照明对应 | 6/6 先行保护通过，25 passed / 1 failed / 1 ignored | 未见实拍净收益，并把既有 neck texture 回归的 16 级差改成 9。没有放宽期望。[patch](candidate2/exact-lighting-q.patch)、[失败](candidate2/face-tests.txt)、[报告](candidate2/replay-report.json) |
| 单调法线 alpha 剖面 | 6/6 先行保护通过 | 新齿略减，但制造完整旧下颌暗弧，更差。[patch](candidate3/monotone-alpha-profile.patch)、[报告](candidate3/replay-report.json) |

![拒绝的切线平均，只模糊凸起](candidate1/portrait-makeup-chin--100-edge-4x.png)

![拒绝的单调剖面，新增完整暗弧](candidate3/portrait-makeup-chin--100-edge-4x.png)

这些候选的完整输出保留在忽略的 test/results，不进入生产。第二候选失败的纹理回归，亦说明“低频更平顺”不能自动等同“细节保持更好”。

## 下一步必须解决的内容

1. 将 face/neck 与 hair/scene 的类别置信度、物理 alpha 和可见层次分别估计。可用有真实 trimap 输入的 matting 模型验证 mixed edge，但先与当前 source 原尺寸轮廓比较，不能只换模型名称。
2. 对隐藏的 neck/hair/scene 分别建立沿有序可见边界的供体对应和纹理延续，禁止 BFS 最近类别或跨材质平均造成耳下色楔。无可靠同材质支持时应保守失败或需要手工供体。
3. 将颈部表面纹理与下颌阴影分开，重建新轮廓的照明；现有二维低频平移不足以消除旧弧和色泥。大幅缩短需要更可靠的结构/光照估计，不能继续靠局部模糊掩盖。
4. 补侧脸、多肤色、胡须、发丝遮挡、高分辨率原照和真实浏览器热运行。当前五个独立人物及768px左右输入仍不足以支撑普适 8 分。
