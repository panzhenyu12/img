# 可选本地 LaMa Worker

使用原 LaMa generator 的公开 TorchScript 导出做 CPU 推理，无需 LLM、DeepAgent 或 API key。手工蒙版也可直接执行。Go 管理 session、图层、MCP 和任务；Rust 管理统一蒙版和最终合成。

## 安装与运行

从仓库根目录：

```bash
make background-models
make background
# 另一个终端
make background-test
make dev
```

独立 Python 3.11（Intel macOS 验证版本 3.11.14）、Torch 2.2.2、NumPy 1.26.4，不替换系统 Python。模型在 `.models/background/big-lama.pt`，约 206 MB，不进入 Git；显式安装时联网，启动不自动下载。默认 bind `127.0.0.1:8091`。Go 使用 `IMG_BACKGROUND_WORKER_URL` 或 `background.worker_url` 配置连接；worker 支持 `BACKGROUND_LAMA_MODEL` 指定固定校验权重路径、`BACKGROUND_CPU_THREADS` 控制 CPU 线程。

也可运行 `sh services/background-worker/setup.sh`：uv 分支复用 `.venv`；没有 uv 时使用已有 Python 3.11 venv/pip。模型下载会恢复有效 `.part` 并在完成后验证 SHA256 与字节数。

## HTTP

`GET /healthz`；`POST /v1/inpaint`。multipart：`image` 为 PNG/JPEG/WebP（按 EXIF 展示方向），`mask` 必填灰度 PNG，`protect_mask` 可选；蒙版必须匹配展示方向尺寸。Go 公开代理 `POST /api/inpaint/model` 字段相同。返回原尺寸 RGBA PNG，全部 alpha、冻结覆盖和实际选区外保持原字节。

| 表单 | 默认 | 范围/含义 |
|---|---|---|
| max_side | 1024 | 256–2048；等比例缩小模型输入 |
| context_mode | full | full / 显式 crop |
| context_padding | 256 | crop 时目标周围 64–1024 像素上下文 |
| mask_dilation | 4 | 0–32；只扩推理洞，最终仍约束原目标 |
| seam_blend | none | none / 可选 poisson（目标不超过 300000 像素） |

目标、冻结与透明 RGB 先排除，缩小时图像与蒙版均使用正面积 BOX footprint；蒙版以浮点 coverage >0 判断，强缩小时单像素未知不能因8位量化消失或以缩放黑边污染已知上下文。空目标不加载模型；非空目标必须有已知、非透明、非保护背景。缺权重或错误 checksum 返回 503，非法尺寸/参数返回 422。

`X-Inpaint-*` 响应头报告模型、CPU、推理/处理时间、进程 lifetime 最大 RSS、目标数与上下文方式。推理时间不含 PNG 网络传输，HTTP wall time 更完整；第一次加载/JIT 优化明显较慢。没有实时性能承诺。多个模型图层可设置 Go `IMG_RENDER_TIMEOUT=5m` 并使用异步任务。

## CLI、测试与权重核对

```bash
services/background-worker/.venv/bin/python services/background-worker/worker.py \
  --input doc/qa/background-lama/street-source.png \
  --mask doc/qa/background-lama/street-mask.png \
  --protect-mask doc/qa/background-lama/street-protect.png \
  --output /tmp/street-clean.png --report /tmp/street-clean.json
```

包装合同测试：`make background-test`。这些 fake-model 测试不评价画质；[QA](../../doc/qa/algorithm-audit-lama.md) 单独记录真实 Torch/HTTP 推理、原图与失败结果。

核对第三方导出与官方 generator：

```bash
services/background-worker/.venv/bin/python services/background-worker/scripts/download_models.py lama-official-checkpoint
uv pip install --python services/background-worker/.venv/bin/python -r services/background-worker/requirements-verify.txt
services/background-worker/.venv/bin/python services/background-worker/scripts/verify_weights.py
```

官方 zip 381 MB 仅此核对步骤需要，运行 worker 不需要训练框架。已验证 989 个参数/buffer 全相等，不表示重训复现。原仓库：[advimman/lama](https://github.com/advimman/lama)；论文：[Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161)。

## 效果边界

原 768×576 街景和缩小 512 的正面积预处理结果已能移除目标人物。复杂人群、建筑直线与遮挡细节仍需逐图检查。旧版本 Lanczos RGB/BOX mask 不一致造成灰石纹伪影，已修复并保留失败证据；`crop` 或过低分辨率可能丢失场景语义，不自动启用。Poisson 只调整梯度/接缝，不恢复建筑结构。浏览器 ONNX 另见 [客户端研究](../../doc/research/lama-client-runtime.md)。
