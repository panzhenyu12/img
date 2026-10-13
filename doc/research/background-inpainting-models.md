# 背景人物/杂物去除：论文、原实现与运行选择

日期：2026-10-13。当前实现不把传统补丁修复称为最新论文算法，也不把 LaMa（WACV 2022）称为2026前沿。先用原始模型权重和真实图证明可运行，再逐项比较更新的候选。

| 算法 | 原论文/公开代码 | 已证实情况 | 本期处理 |
|---|---|---|---|
| Criminisi exemplar | [CVPR 2003](https://doi.org/10.1109/CVPR.2003.1211538) | 自身确定性 CPU Rust 近似；条纹/棋盘精确，街景有接缝 | 手工、无模型后备；不宣称原论文完整复现 |
| LaMa | [WACV 2022 / arXiv2109.07161](https://arxiv.org/abs/2109.07161)、[advimman/lama](https://github.com/advimman/lama) | 原 generator989 tensor与官方权重全相等；CPU Torch/HTTP真推理；Carve512 ONNX与同输入Torch误差<0.004/255 | 已接本地服务；客户端另行验收 |
| MAT | [CVPR 2022 / arXiv2203.15270](https://arxiv.org/abs/2203.15270)、[fenglinglwb/MAT](https://github.com/fenglinglwb/MAT) | 官方提供Places512/FFHQ权重；README说明发布权重重训且可能与论文表格不同；推荐旧Python3.7/Torch1.7.1/CUDA11环境 | 尚未跑；不能把官方CUDA例子当成本机CPU已经验证，也不直接套浏览器 |
| BrushNet / BrushNetX | [ECCV2024 / arXiv2403.06976](https://arxiv.org/abs/2403.06976)、[TencentARC/BrushNet](https://github.com/TencentARC/BrushNet) | SD1.5/SDXL + 独立BrushNet权重；官方repo公布下载入口，BrushNetX参考BrushEdit | 更大扩散模型候选，未在本机下载/推理，不能报质量和延迟 |
| PowerPaint | [ECCV2024 / arXiv2312.03594](https://arxiv.org/abs/2312.03594)、[open-mmlab/PowerPaint](https://github.com/open-mmlab/PowerPaint) | 官方HF `JunhaoZhuang/PowerPaint-v1/PowerPaint_v2/PowerPaint-v2-1` 多版本权重和任务prompt | 复杂语义去除候选，需固定版本、GPU/CPU实测、原图结构与保护区验收 |

以上后三者“开源可下载”不等于本期已经实现。CPU上的扩散图模型耗时/内存仍未知，不能只凭名气直接替换已测试代码。MAT的官方旧依赖可通过独立worker隔离，CPU是否可用及checkpoint兼容需要实跑确认。

LaMa原TorchScript的FFT在当前PyTorch2.2.2 opset17直接导出失败；Carve通过矩阵变换表达Fourier运算绕过。现有ORT-Web1.30包含DFT/Einsum/MatMul实现，浏览器可评估已有转换，不能默认所有原PyTorch模型可自动放进WASM。详见 [客户端核查](lama-client-runtime.md)。

每种模型接入统一的“原图 + canonical目标/保护PNG → 原尺寸候选PNG”接口，最后仍由Rust约束目标、冻结、alpha、opacity与blend。Go/AI/MCP只调用Operation，模型服务独立于LLM。增加新模型前必须记录原仓库/论文、checkpoint来源与哈希、预处理、runtime/设备、真实耗时、明确失败图和自然度对照；不把通过像素保护测试当成画质通过。
