# ZIM 原照边缘 alpha 实测：可研究，不接受为可动面部的物理 alpha

日期：2026-10-14。官方 ICCV 2025 Highlight ZIM 的 vit-b ONNX 已真实下载并在本机 CPU 执行。五组提示 × 四个候选 alpha 都已保存、查看。**原始 ZIM alpha 未通过可动面部合同**：虽然能大体选出脸，耳朵、部分颈部与饰品仍被包括；用语义 mask 再裁掉它们属于本项目适配，也不能自动保证边界自然或修复形变场。

## 输入、官方模型和隔离

一张独立源图：用户原照 canonical 1024²，SHA256 `5f15e92b14c44a468d924af5ac985a7cc7dc1c87d8ab235f2c94b2e88b173c73`；原照检测 graph SHA256 `9ff494d8434a751833d98fe60f598b179d66813531e0a581013cbbcac33f717c`。检测、提示与语义 reference 全来自原照，没有重检测编辑结果。

官方 [代码提交](https://github.com/naver-ai/ZIM/tree/392e613c8a3d395ce4dda3b51edb6dabc1fd0004)、[论文](https://arxiv.org/abs/2411.00626)、[公开权重 revision](https://huggingface.co/naver-iv/zim-anything-vitb/tree/883e9bbb15ca35a6afe16b72ae7e55cc13efbc78/zim_vit_b_2043)。模型实际字节与官方 LFS SHA 相符：encoder 360,680,416 字节 / `6123ea10722dd43a0f8bbb96cd6a0532d8a65e2b1e8339f0cd6593d494807809`，decoder 19,330,176 字节 / `a99d14bac3dc28b17809020b1711b1991bd5b7eb18f678c3624ce508748cfa11`。总计 **380,010,592 字节**；无账户、无私有或替代权重。

执行未修改的官方 encoder/decoder、Zim、predictor 与 ResizeLongestSide；默认 CPUExecutionProvider，线程从官方16设为3。原 preprocess、point padding/label、attention mask、四个 mask、原图尺度 postprocess 与 sigmoid 皆执行。只跳过 unrelated 包初始化，不导入自动 mask/SAM/demo。Python3.11.14、Torch2.2.2、NumPy1.26.4、ONNXRuntime CPU1.19.2；新包只进 `.run/zim-deps`，生产后台 `.venv` 未改变。源码逐文件 hashes 见 `experiments/zim-20261014/vendor/manifest.json`。

## 实际时间和内存

macOS26.6.2 / x86_64，CPU，3线程。session/model 初始化 **2,141.78ms**；第一次原图 set_image **7,966.41ms**，重复编码 **9,021.44ms**。每组 decoder+postprocess 首次/第二次如下。原照编码可缓存；重复编码并不一定更快。进程峰值 RSS **5,533.22MiB**，包含 session、Torch、features、mask与对照生成；不是 GPU VRAM 或纯权重内存。

| 提示组 | 正/负提示范围 | decode首次/重复 ms | 官方score最高索引 | 该alpha前景像素 |
| --- | --- | ---: | ---: | ---: |
| 单脸正点 | 鼻区一个正点 | 755.03 / 912.45 | 2 | 360,220 |
| 脸解剖正点 | 鼻、双眉、唇、双眼、额、双颊、下巴10点 | 495.46 / 504.63 | 2 | 155,868 |
| 正负点 | 上述10正点+耳/发/颈7负点 | 490.42 / 480.94 | 2 | 108,726 |
| 原脸框 | 原照可动脸mask包围框 | 517.00 / 533.56 | 2 | 107,584 |
| 加密正负点 | 10正点+13个耳/发/颈/饰品负点 | 880.18 / 755.68 | 2 | 107,382 |

每组实际执行两次。二十个输出均保留，并非只展示最好一张。最高 index 基于官方 predicted IoU score；**这个预测分数不是独立真值 IoU、产品质量分或8分验收**。本次没有浏览器/WebGPU、CUDA或批处理性能结论。

## 原照冻结部位失败

以下都按原照检测 coverage≥0.85 的内部区域监测，统计 alpha≥0.5 的占比；没有根据输出重新定义耳/颈位置。表内为官方score最高候选。

| 提示组 | 原耳区仍为前景 | 原颈区仍为前景 | 原发区仍为前景 | 与旧physical alpha阈值IoU |
| --- | ---: | ---: | ---: | ---: |
| 单脸正点 | 93.70% | 99.79% | 0% | 0.278603 |
| 脸解剖正点 | 92.05% | 73.30% | 0% | 0.628636 |
| 正负点 | 91.25% | 3.23% | 0% | 0.883847 |
| 原脸框 | 92.00% | 0.86% | 0% | 0.897664 |
| 加密正负点 | **90.95%** | **0.99%** | 0% | 0.896192 |

加密组仍保留 **5,355 / 5,888** 个耳内部像素、618 / 62,353 个颈内部像素、385 / 3,820 个饰品内部像素。耳均值 alpha 0.904708；左耳负点 `(337,325)` alpha **0.981265**，右耳负点 `(676,341)` alpha **0.999729**。其余三个候选也不满足两个耳负点，不能通过换index把耳失败隐藏掉。发内部 55,764 像素没有 alpha≥0.5，均值0.000270；这只证明本图头发内部基本排除，不证明发际软边是正确物理alpha。

双眼、双眉与唇的原照多边形内部 alpha≥0.99 为100%，说明模型保留这些脸部区域；仍不说明其余轮廓权限通过。全图 `skin` reference 含肩/颈，不能把全图皮肤保留率低误报成面部皮肤缺失。

![五组中加密提示的四个原始alpha对应cutout](face-dense-signed-prompts-four-mask-review.png)

## 与当前 coverage/physical 的实际对照

对照只改变 alpha 显示，不重新生成照片RGB。六列依次为原照、ZIM原始最高score alpha、语义排除后的ZIM、BiSeNet face、BiSeNet可动脸coverage、归档旧physical alpha。现有 BiSeNet `face` 本身包含更多头/耳区域，不能当作可动脸层直接采样。

![原照与各alpha](face-alpha-comparison.png)

![同一原照RGB的alpha cutout](face-cutout-comparison.png)

ZIM原始的脸/颈连接有轻微软残留，耳根明确保留整耳。语义适配单独计算 `raw * (1 - max(original hair, ears, neck, accessories coverage))`，不是官方推理结果；与 BiSeNet可动脸/旧physical的阈值IoU为0.950366 / 0.949520，MAD为0.005065 / 0.005165。**这些仅表示与reference一致程度，reference并非ground-truth alpha，不能称抠图准确率95%**。

适配之后 frozen内部没有alpha≥0.5，但耳区mean alpha仍0.004068，裁切的耳缘可见细残留；发际被官方 alpha软化，局部源细发丝表现与旧physical不同，没有真值无法断言谁更好。当前缺陷也不能用“原脸大体取得”宣告解决。

![原生像素右耳对照，nearest2x](right-ear-comparison-2x.png)

![原生像素脸颊对照，nearest2x](right-cheek-comparison-2x.png)

## 判定与下一步边界

- 真实可复现：官方权重hash、CPU执行路径、原照feature缓存分离、五组原始提示输出与时间。可保留为用户交互抠图的研究工具。
- **不接受直接作为 movable-face physical alpha**：原照耳与部分颈/饰品权限失败；后续若用语义分割约束未知边界，需要新实验，不直接以本次适配进入生产。
- 未执行任何新alpha的瘦脸warp。更好的source alpha只改善源边界/修改权限；**不能修复旧形变场alpha=1→0的位移断裂、Jacobian折叠或耳根/背景的重建**。不得把本实验当作脸型算法已达标。
- 单张源照片（带“AI创作”水印），没有ground-truth alpha、跨照片泛化或身份评价，不打8分。实验已冻结；本次不继续下载其他大权重，不改生产Rust/WASM/后台服务。

完整浮点20候选在 `test/results/zim-20261014/*/alpha-all-f32.npy`；原始加密组浮点SHA256 `54320ceb45851e89e3c1e6e533edc5664a54a2cdd65fe94c294b008610949729`。归档 [experiment-report.json](experiment-report.json)、[review.json](review.json)、[prompts.json](prompts.json) 记录实际模型/输出/脚本hash与fixed像素profiles；归档raw/适配f32为对应裁切独立证据。复现脚本与依赖命令在 `experiments/zim-20261014/README.md`。
