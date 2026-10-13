# LaMa 客户端运行与 ONNX 导出核查

日期：2026-10-13。结论：客户端模型路径已实现，已有可核查的 ONNX 转换；本地 Python ORT 的真实推理与原 TorchScript 数值接近。浏览器也已独立完成街景、人群、建筑三个样片，当前实际后端为WASM；WebGPU实际执行与性能仍未验收。详见[FUNC-016](../func/FUNC-016-local-model-inpaint.md)，不能用Python成功代替浏览器证据。

## 具体模型

来源：[Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX)，revision `c3c0c9e468934d62e79c329e35d82dd09ff8c444`，文件 `lama_fp32.onnx`。本工程缓存 `apps/web/public/models/lama-fp32-512.onnx`，208044816 字节，SHA256 `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6`。该文件是转换推理模型，不能称新的前沿算法。

实读 ONNX protobuf，而非依据示例猜测：

| 接口 | 实际定义 |
|---|---|
| opset | ai.onnx 17 |
| image | float NCHW `[batch,3,512,512]`，RGB 0..1 |
| mask | float `[batch,1,512,512]`，1=需要填补 |
| output | float NCHW，实际 `[1,3,512,512]`，RGB **0..255** |

尾部先计算 `mask * generator + (1-mask) * image`，再乘常量 **255.0**、Clip 到 0..255。输出再次乘 255 会造成截断/过曝，不能照搬服务端的 0..1 输出转换。证据：[client-onnx-inspection.json](../qa/background-lama/client-onnx-inspection.json)。

## 为什么原 TorchScript 不能直接交给 ORT Web

TorchScript 由 PyTorch Runtime 执行，浏览器 ORT 读取 ONNX；Rust WASM 也不会自动提供 Torch 运算符。LaMa FourierUnit 使用实数 FFT、复数实虚部变换和逆 FFT。当前 Intel Mac/Python 3.11/Torch 2.2.2 的真实 `torch.onnx.export(...,opset_version=17)` 对原模型失败：`UnsupportedOperatorError: aten::fft_rfftn`。这只证明当前导出组合不支持，不证明所有未来 exporter 都不可转换。

Carve 转换将 Fourier 运算拆为矩阵/逐元素计算。本文件没有 ONNX `DFT` 节点；实际包含 216 `MatMul`、216 `Einsum`、Cos/Sin、reshape/slice/pad 等，外加原卷积。其大量节点与约 208 MB 权重影响加载、图优化、内存和计算成本。

本工程安装的 **onnxruntime-web 1.30.0** 源码 registry 和 operator 文档已包含 WebGPU `DFT`（opset 17+）、`Einsum`（12+）、`MatMul`；因此不能笼统说“WebGPU 无 FFT/DFT”。真正问题是 PyTorch→ONNX 的表达、所选模型每个 operator 的后端支持、数值误差和实测内存/性能。DFT kernel 对合适长度有 FFT 实现，对其他长度可能使用更昂贵的直接计算；但此 Carve 文件实际走矩阵转换，不会自动改用 DFT kernel。

## 同输入真实数值核对

同一街景使用一致 BOX 图像/蒙版缩小至 512×384，并对称 pad 到 512×512；目标、冻结人物与 alpha=0 RGB 都排除，两个 runtime 输入张量完全相同。TorchScript 输出转换为 0..255 后与 Carve ORT CPU 输出比较：

| 项目 | 实测 |
|---|---:|
| Torch CPU，4 threads | 6914.79 ms |
| ONNX Runtime Python 1.20.1 CPU，4 threads | 5726.48 ms |
| 全图最大误差，0..255单位 | 0.003311 |
| 全图平均误差 | 0.000004095 |
| 洞内平均误差 | 0.000125387 |

这说明该转换在这张图、固定 512 输入和该 CPU backend 上接近原模型；不是浏览器耗时，不代表所有尺寸/显卡严格一致，也不能保证每个最终 uint8 都一致。224 个大 initializer 中 147 个与原 Torch 权重按字节精确匹配，其余没有全部逐项归因；不能以此替代完整转换和数值验收。服务端自身的 989 个权重/buffer 与官方 checkpoint 则已全部精确验证。

证据：[client-onnx-parity.json](../qa/background-lama/client-onnx-parity.json)、[direct-onnx-export.json](../qa/background-lama/direct-onnx-export.json)。原始 tensor candidate 会重绘推理保护洞，只有公共 Rust 合成后才是最终修图结果；不能把 raw candidate 当成完成图展示。

## 实现约束

客户端 `execution=client` 与服务端 `execution=server` 写入同一 Operation；客户端不自动上传或切换服务端。两条路径都使用同一 Rust target/protection 光栅化与最终合成。图像缩小与蒙版的采样 footprint 必须一致且为正面积核；本次服务端发现 Lanczos RGB 配 BOX mask 会把黑洞颜色带入已知背景，导致石纹伪影。另外uint8 BOX在强缩小可量化掉单像素mask，已改为float覆盖>0；客户端面积交集>0不量化mask。单测验证“外部没变”抓不住这种洞内错误，真实图必须复查。修复与样片证据见[LaMa QA](../qa/algorithm-audit-lama.md)。

客户端固定512模型要等比例缩小并补到512正方形，不能拉伸照片；原分辨率 target/protection/alpha 留给最终合成。无需再做 DFT/MatMul 的自制 WASM 算子，优先调用现有 ORT Web。可选择 WebGPU（浏览器提供时）或 WASM；WebGL 缺完整算子覆盖，不作为此模型的默认路径。

浏览器还需独立验证：首载网络与缓存、约208 MB资源、WebGPU/WASM fallback、等待和取消、模型缺失、正方形 padding、蒙版边缘、保护和 alpha、前后图层顺序、多张实际照片与大洞。模型资产可本地缓存；只有模型已加载并缓存后才讨论离线运行，不能把首次下载算作纯离线能力。

## 参考

- [原论文 LaMa](https://arxiv.org/abs/2109.07161) 与 [原代码](https://github.com/advimman/lama)。
- [ORT Web WebGPU operators](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md)，本地判断基于已安装 1.30.0 文件。
- [ONNX DFT 算子定义](https://onnx.ai/onnx/operators/onnx__DFT.html)。
