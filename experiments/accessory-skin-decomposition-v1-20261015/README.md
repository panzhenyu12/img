# 用户耳饰边缘隔离实验（2026-10-15）

本目录是用户原图 `test/results/user-face-contour-20261014/1024/canonical-source.png` 的隔离诊断，使用 exact V36 `user-face-contour-{100,-100}-full`。没有改 Rust/WASM、服务、生产默认或既有 hair 实验。

**最终严格门禁版本为 `final-gated-v5/`。** 原 `neural-backing-v5-virtual-scene/` 以神经估计可用替代了旧局部 scene 距离不足门禁，因此不能声称原 insufficient 域仍未写入。最终派生以冻结 v2 输出为底，只写 source-support ∩ v2 原 write mask，并排除 v2 原 unknown、insufficient；不再写 transported/destination 边。4 组全部门禁断言通过，视觉仍拒绝。报告 `final-gated-v5/report.json` 的 `accepted=false`。

## 结论

首轮 `outputs/`、软保护桥接 `outputs-v2/`、LaMa 全上下文 `neural-backing-v3/`、语义排除上下文 `neural-backing-v4-clean-context/`、虚拟背景上下文 `neural-backing-v5-virtual-scene/` 全部保留。最终 scene-only 分支停止，未达到可用效果：+100 候选仍有明显竖向肤色带/断续双边，且 LaMa 在未知区域会生成脸/耳朵结构；-100 黑楔有所减少但仍有锯齿/尖口。没有把某个方向的局部改善当成算法通过。

`neural-backing-v5-virtual-scene` 是最后受控 scene 实验；随后又做了唯一允许的输入级 hybrid：

- `hybrid-crop2/` 固定 component 1 的 full corrected alpha，只把 component 2 的 4x `combined-point` local alpha（official index 2）贴回原图 crop。
- source decomposition、81 个独立 source-skin support、V36 inverse、unknown/insufficient refusal 与金属前景全冻结。`hybrid-crop2/report.json` 记录 source/graph/inverse/crop-audit SHA，`test_hybrid.py` 通过所有字节和越域保护断言。
- +100 仍有竖向肤色带/断续双边；-100 仅局部变化，黑楔/锯齿仍在；`accepted=false`。因此 4x 局部分辨率 alpha 没有达到接入标准，拒绝该候选。

`neural-backing-v5-virtual-scene` 是最后受控 scene 实验：

- 固定 corrected accessory alpha（两个原有组件，官方候选 index 2）、source/graph、V36 inverse map、金属前景颜色和 alpha。
- 先用 subject mask、physical source alpha、face/skin/neck/hair/ears/accessories 语义及 2px exclusion 得到已观测 scene；未知上下文只用最近已观测 scene RGB 做虚拟输入，再将 LaMa mask 限定为 source support 的 2px 局部带。
- 原 v5 写入限 source-supported/transported support/connected coverage，unknown 不写；以神经 backing 可用代替旧局部 donor 距离门槛。`final-gated-v5` 才重新强制原 insufficient 拒绝，并仅修改原 source 支持域。两者严格区分，所有 128/256、±100 结果保留。
- 生成结果仍失败，证明问题不是单纯把更大的 LaMa crop 或颜色先验换掉即可解决。

## 实际证据

`neural-backing-v5-virtual-scene/report.json`：

- 4 组结果（128/256 × +100/-100），source SHA `5f15e92b14c44a468d924af5ac985a7cc7dc1c87d8ab235f2c94b2e88b173c73`，graph SHA `9ff494d8434a751833d98fe60f598b179d66813531e0a581013cbbcac33f717c`。
- 逆映射分别固定为 +100 `9423e4d...`、-100 `4770a03...`；每行也记录完整 SHA。
- 128/+100 写入 969 像素、输出变化 950；128/-100 写入 1363、变化 1112；256 与对应尺寸相同。unknown new coverage 均 0 是写入门禁结果，并不表示前景/背景语义真值已知。
- `opaqueMetalChanged=0`、`outsideWritesChanged=0`、`alphaRefined=false`；测试断言保持金属核心、输出 alpha、逆映射和 V36 几何不变。
- `virtualContextIsReconstructionHypothesis=true`：虚拟 scene RGB 不是观测真值。

原尺寸对照：

- `neural-backing-v5-virtual-scene/128/100/component-1-1x.png`
- `neural-backing-v5-virtual-scene/128/100/component-2-1x.png`
- `neural-backing-v5-virtual-scene/128/-100/component-1-1x.png`
- `neural-backing-v5-virtual-scene/128/-100/component-2-1x.png`
- 对应 256 目录，以及 `face-1x.png`、conditioning、scene、write/unknown masks。

## 已识别的错误来源

1. 旧 V36 的 accessory frozen 域包含连通皮肤和耳饰内部背景；只修 alpha 而继续跳过 frozen 会保留整片台阶。
2. 清理旧条和写入移动后的新轮廓必须是同一个 source/destination 合同。首轮若仅让新 destination 写入，+100 会形成“双边”。
3. corrected alpha 的低值并不代表孔洞是皮肤；独立 audit 显示 component 2 model-excluded 723 像素中只有 81 有源 skin 支持、642 unknown，component 1 中 141/658。金属高光核心必须保持原 RGB/alpha。
4. LaMa 全上下文会把未知耳饰孔洞补成脸/耳朵；排除 semantic foreground 后仍会生成错误的人形结构。虚拟 scene 先验只能作为明确假设，不能当成真实背景。
5. alpha 无界放宽会抹掉细发真实覆盖；bounded ±0.15 因果实验在 `experiments/alpha-cap-causal-20261015/`，不能接入此功能。

## 重跑与范围

环境：`services/background-worker/.venv/bin/python`。脚本默认拒绝覆盖已存在目录；重跑需使用新输出目录并更新 hash。

```sh
services/background-worker/.venv/bin/python experiments/accessory-skin-decomposition-v1-20261015/run.py
services/background-worker/.venv/bin/python experiments/accessory-skin-decomposition-v1-20261015/run_v2.py
services/background-worker/.venv/bin/python experiments/accessory-skin-decomposition-v1-20261015/neural_backing.py
services/background-worker/.venv/bin/python experiments/accessory-skin-decomposition-v1-20261015/neural_backing_clean_context.py
services/background-worker/.venv/bin/python experiments/accessory-skin-decomposition-v1-20261015/neural_backing_virtual_scene.py
services/background-worker/.venv/bin/python experiments/accessory-skin-decomposition-v1-20261015/final_gated.py
```

这些是诊断和失败记录，不是 8 分实现。下一步应先得到可靠的饰品/皮肤深度与孔洞前景支持，再设计清除与 destination 写入的双向原子测试；继续调 RGB 阈值或扩大 LaMa mask 会掩盖归属错误。

## 后续 model-free scene donor 审计

`scene-donor-v1/` 与 `scene-donor-v2-local/` 均实际生成 ±100 原尺寸候选并拒绝。金属/耳朵、alpha、inverse 与区域外不变量均通过，但 +100 旧肤色条和 -100 黑楔仍在，目标边界梯度恶化。详细方法、统计与独立测试见 [scene-donor-README.md](scene-donor-README.md)。这次实验指出应修正缺失的 source skin support；不能只更换稀疏旧写入域中的背景颜色。
