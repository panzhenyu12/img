# Inpaint4Drag 2025 双向 warp 隔离实验

固定官方提交 `bfb0fafd456ac01ee23ade0cc04b64c9781eaa87`，仅复用未修改的 `utils/drag.py` 为 `vendor_drag.py`。来源：[Visual-AI/Inpaint4Drag](https://github.com/Visual-AI/Inpaint4Drag/tree/bfb0fafd456ac01ee23ade0cc04b64c9781eaa87)，论文 ICCV 2025 [2509.04582](https://arxiv.org/abs/2509.04582)。未导入官方 `ui_utils.py`：它在import时自动下载/加载SD+LCM并要求CUDA。

实验使用用户原图的canonical1024与只在原图检测的graph。每个±30/±60及+100的实际目标轮廓由冻结的trimap-candidate native导出；source/target已经是原图绝对坐标，`origin`只用于debug mask裁剪。根据官方整数UI点击把点对round，浮点原目标另存。

先配置隔离OpenCV，不修改后台服务环境：

```sh
uv pip install --target .run/inpaint4drag-deps --no-deps opencv-python-headless==4.8.1.78
services/background-worker/.venv/bin/python experiments/inpaint4drag-20261014/run.py --lama
```

依赖复用后台Torch2.2.2、NumPy1.26.4、Pillow、SciPy；CPU3线程。输出默认 `test/results/inpaint4drag-20261014`，已有结果拒绝覆盖。`--out`指定新目录，`--amounts`可以选子集，`--binary`仅应指定能导出`FACE_DEBUG_DIR/geometry.json`的隔离native，禁止重建公共WASM。

各case保存原始官方warp、目标mask、露底/轮廓补全mask、施加保护后的warp、已有LaMa的局部补全基线，以及原生尺寸与nearest2x对照。LaMa不是论文的SD+TAESD+LCM；论文V100的0.3s不能套用于本机CPU。所有图/graph/二进制/源码/模型哈希和实际耗时写入report。

裸官方输出与额外保护合成分别报告：头发/耳朵/饰品>=0.5区域、远背景（source与target轮廓并集+5px）、alpha变化。眼区另报告副作用，未靠眼冻结掩盖算法修改。扩张需要写入目标外轮廓，因此“原脸以外背景全固定”不作为自相矛盾的合同；远背景必须固定。

已知官方限制：top4 inverse-distance插值、逆映射最多100 reference、整数round、只处理外轮廓（内洞会吞掉），未证明保直/无折叠。合同通过仅证明输出权限，不能证明自然度；未验收不接生产。
