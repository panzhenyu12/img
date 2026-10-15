# Gemini 几何归因消融：用户原照，2026-10-14

用户原照的主要黑裂缝没有被“去缓存、平滑切换、扩大几何过渡带”解决。当前证据支持移动皮肤与固定耳根缺少连接约束；它没有证明几何场的所有问题都已排除，也没有证明 BFS 的材质接触面就是主要黑缝根因。

## 输入、冻结与范围

- 原图 `test/images/user-face-contour-20261014.jpg`：1024×1024，独立输入 1 张，含“AI 创作”标记。768 是重采样派生，不是独立相机照片验证。
- 两尺寸分别 fresh 检测原图一次，输出不重检测。本报告消融只使用 1024 原尺寸。
- 浏览器解码输出 canonical PNG，避免 Rust 与浏览器 JPEG 解码差异。完整 mask、36 个 landmarks 和 graph 在 `test/results/user-face-contour-20261014/1024/`。
- 冻结 baseline 为该专项 `baseline-image-render`，SHA-256 `620f9b815f55c62e5374bef46d4e01c670e1f0a0308c02b84c82046f1b37660f`；生产 face.rs SHA 为 `4d9e77ff91f94682649283e73f8c6c3d03e62da49d82c1655e24f9ff476c38a2`。
- 每个候选只改一处几何路径，所有生产源码、公共 native/WASM 均未改变。完整 binary/source/graph/model/input 哈希见 `provenance.json`、`variants.json`、`variant-results.json`。

## 单因素消融

| 候选 | 改动 | +60 与 baseline 改变像素 / 最大通道差 | −60 与 baseline 改变像素 / 最大通道差 | 两侧主要黑点 |
| --- | --- | --- | --- | --- |
| grid-exact | 所有像素取精确 contour/MLS，取消 8.657px 网格/精确硬切换 | 824 / 3 | 623 / 1 | RGBA 完全不变 |
| grid-smooth | 在 3–8.657px 连续混合 grid 与 exact | 361 / 4 | 112 / 1 | RGBA 完全不变 |
| band | contour/MLS 插值范围从 3px 扩大到 8px | 6,538 / 131 | 2,298 / 38 | RGBA 完全不变 |

每个候选的固定耳朵/首饰/头发半覆盖像素变化为 **0**，alpha 变化为 **0**。这些是合同验证，不是自然度评分。人工原尺寸及 2 倍 nearest crop 复核：+60 的宽暗缝和阶梯耳根仍在；8px 过渡改变了更多局部肤色与边缘，却没有修复主要耳根断接。−60 的耳根拼接线与下颌阶梯也未取得稳定净收益。

`right-ear-root-native-pixels.png` 保留未重采样像素；`*-right-2x.png` 仅用 nearest 放大。没有通过模糊图或缩略图掩盖锯齿。

## 纯诊断 trace 的像素因果链

另外复制 baseline 源码，只导出中间值，不修改算法。trace 的 +60/−60 输出 PNG SHA 与冻结 baseline **完全相同**，见 `trace-report.json`。mask/材质以灰度导出，精确浮点 probe 保留在 JSON：

| +60 原尺寸位置 | 原 RGB / 物理 alpha | 逆采样新 alpha | 冻结 | 实际路径 | 重建材质 / 背景 RGB | 回退 |
| --- | --- | --- | --- | --- | --- | --- |
| 左 `(361,335)` | 149/112/94，1 | 0 | 否 | exact，距 target 0.310px | hair，67/49/44 | 否 |
| 右 `(648,384)` | 181/134/114，1 | 0.001971 | 否 | exact，距 target 6.409px | scene，59/48/46 | 否 |

两点原 face/skin/faceContourMask 均为 1，ear/accessory/hair/neck 均为 0。它们先被逆采样判为空洞，再填成暗色底层，与保持原位的耳根断开；它们不是被硬冻结直接拷贝出的黑像素。右侧邻接 `(654,384)` 与 `(655,384)` 确为冻结点，并保持原像素。

两点均在 jaw clipping 的 0.76 高度门槛之前，因此该函数的 distance8/support0.999 条件根本不进入这些耳根黑点路径。不能把下巴阈值存在解释成上半脸耳根裂缝的原因。

耳根 ROI 存在 267 个不同 reconstruction material 的接触像素。左、右主要黑点分别是 label2、label0，距材质切换约 2、2.236px；耳根 ROI 中 estimated_background fallback **0**，全 crop 仅 13 个 fallback。当前证据支持材质重建可能产生额外色缝，**不支持直接宣称主要黑缝由两股 BFS 在这里相撞或内部孤岛回退造成**。证明该归因仍需单独材质约束消融，不能以一个 `layer[n] == layer[i]` 判断代替实验。

−60 同样 probe 的新 alpha 为 1，无 hole，并输出正常皮肤；主要暗缝机制有方向差异。扩张时仍有耳根几何阶梯，不应把 ±60 当成同一个重建故障。

## 对 Gemini 建议的结论

“固定耳根与移动皮肤断接”有直接证据，但移除固定合同不是所需解法。应保持耳朵、首饰、头发、用户冻结像素，约束邻接皮肤的位移及连接，并检查覆盖与颜色连续性。Soft frozen、跨材质混色、所有 clipping 阈值变 soft 尚未在本实验验证，不能直接作为修复结论。

本轮仅排除网格硬切换及单纯扩宽 3→8 对**这张图主要黑缝**的解释。其他照片、几何场折叠、最近线段变化、neck lighting 和低 alpha 边缘仍需要独立验证。

## 复现

先在对应源码状态运行 `node scripts/diagnose_user_face_contour.mjs` 生成 fresh 输入（若已有目录，脚本拒绝覆盖，可传新 output 目录），再运行 `services/background-worker/.venv/bin/python scripts/replay_user_face_contour.py` 得到冻结 baseline 30 项。重新检测产生新 graph 时，需记录新哈希，不能与旧报告混为同一 graph。

几何消融 patch 位于本目录：`grid-exact.patch`、`grid-smooth.patch`、`band.patch`。各自应用到**独立复制的** image-core/src/face.rs 后，用独立 `--target-dir` 构建 native，再回放冻结的 `1024/baseline/faceWidth60.json` 与 `faceWidth-60.json`。`variants.json` 保留准确源文件哈希；源码哈希不匹配时必须先恢复该基线，不能在后续生产修复上套用同一结论。

完整未裁切输出、中间 alpha、frozen、holes、material、fallback 和 reconstruction 图存忽略目录 `test/results/user-face-contour-20261014/gemini-geometry-ablation/`；本目录只保留报告、patch 和代表图。
