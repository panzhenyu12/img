# ZIM ICCV 2025 原照 alpha 隔离实验

固定官方代码 `392e613c8a3d395ce4dda3b51edb6dabc1fd0004`；`vendor/manifest.json` 记录每个未修改源文件 URL/hash。推理使用官方 encoder、decoder、Zim、predictor、ResizeLongestSide；仅用 namespace bootstrap 跳过与推理无关的 automatic-mask/SAM/matplotlib 初始化，encoder/decoder 线程设为 3。没有修改生产代码、公共 WASM 或后台 `.venv`。

公开 HF revision `883e9bbb15ca35a6afe16b72ae7e55cc13efbc78` 下 `zim_vit_b_2043/` 的 encoder.onnx/decoder.onnx 放在忽略缓存 `.models/zim-20261014/`。真实字节/hash见研究文档与运行report；runner 在加载前强制验证，不含自动下载或伪造权重。

隔离依赖与复现：

```sh
uv pip install --python services/background-worker/.venv/bin/python --target .run/zim-deps --no-deps onnxruntime==1.19.2 coloredlogs==15.0.1 flatbuffers==24.3.25 humanfriendly==10.0 protobuf==5.29.5 packaging==24.2 torchvision==0.17.2
uv pip install --python services/background-worker/.venv/bin/python --target .run/inpaint4drag-deps --no-deps opencv-python-headless==4.8.1.78
services/background-worker/.venv/bin/python experiments/zim-20261014/run.py --out test/results/zim-new-run
```

复用后台 Python3.11/Torch2.2.2/NumPy1.26.4/Pillow/SciPy，但所有新增包进入 `.run` target。默认官方 requirements 的 `onnxruntime-gpu` 未混装；用 CPU runtime 1.19.2 运行官方 CPUExecutionProvider 路径。

固定原照 canonical1024；原图检测 graph 不变，不重检测编辑图。五组提示：单鼻正点、十个脸部正点、增加耳/发/颈负点、原脸框、再加边缘/饰品负点。每组保留四个官方 alpha；两次 encoder set_image、两次每组decoder + postprocess计时；一次首次模型session加载与进程峰值RSS单列。`alpha-all-f32.npy` 是完整浮点证据，PNG只是量化展示。

`review.py` 读取默认结果，创建 `doc/qa/zim-20261014`，已有归档拒绝覆盖。单独保存原始ZIM score argmax与 `alpha*(1-frozen)` 的语义排除适配；不把后者称官方结果。原照 BiSeNet/旧 physical 仅是 reference，并非 ground-truth alpha。matting 也不会修复形变场 `alpha=1→0` 的位移断裂。本次没有执行新alpha的脸型warp，没有浏览器/服务端集成，没有跨照片质量验收。
