# FUNC-016 客户端与本地服务端 LaMa 填补

状态：服务端已实现，合同与部分真实图已验证；复杂场景自然度持续验收。客户端已实现且真实浏览器图层/合成已验证；效果与性能仍未达到完整验收。

关联需求：`REPAIR-004` 背景路人去除、`REPAIR-005` 杂物清理、`REPAIR-006` 主体保护。传统手工填补见 [FUNC-014](FUNC-014-background-person-removal.md)。

## 用户行为

在修复工具中手工圈出目标，选择模型填补；需要保留的人或物体可使用冻结区。模型结果创建独立 `repair.inpaint` 图层，支持透明度、关闭、撤销、图层重排和导出。无需聊天、LLM、API key 或 DeepAgent。

客户端模式在浏览器推理；服务端模式调用配置的本地 LaMa Worker。执行方式写入 Operation。服务端导出遇到启用、非零透明度的客户端节点必须报错，建议浏览器导出或显式切服务端；禁止静默改用服务器。模型缺失或执行失败也不自动降级。用户仍可选择传统 exemplar、仿制和修复工具。

## Operation 类型与参数

```json
{
  "id": "remove-extra-person",
  "type": "repair.inpaint",
  "enabled": true,
  "params": {
    "method": "lama",
    "execution": "server",
    "maxSide": 1024,
    "contextMode": "full",
    "contextPadding": 256,
    "maskDilation": 4,
    "seamBlend": "none",
    "mask": {
      "shape": "polygon",
      "points": [{"x":0.75,"y":0.35},{"x":0.86,"y":0.36},{"x":0.86,"y":0.8},{"x":0.74,"y":0.8}]
    },
    "opacity": 100
  }
}
```

`execution` 默认 `server`，允许 `client/server`。服务端 `maxSide` 默认 1024，范围 256–2048；客户端固定 512 模型，不能要求超过 512。`contextMode` 默认 `full`，显式 `crop` 才按目标包围盒加 `contextPadding`（64–1024）裁剪；`maskDilation` 默认 4，允许 0–32；它扩展模型推理洞，不扩展最终可修改区域。`seamBlend` 默认 `none`，服务端可显式选 `poisson`；客户端目前仅 `none`。

`mask/protectMask/region/opacity/blendMode` 使用公共协议。Rust 将选区与保护区域光栅化为原分辨率二值 PNG；原图 alpha=0、任意非零冻结覆盖、禁用节点和 opacity=0 均不能进入实际目标。最终合成仍使用原始软蒙版与公共 opacity/blend，应用一次。

## 前端实现

`apps/web/src/core/render.worker.ts` 按图层顺序拆分编辑图，在模型节点之间继续调用同一 Rust WASM 核心。服务端模式将公共蒙版 PNG 发送到 Go `POST /api/inpaint/model`；客户端模式使用 `clientInpaint.ts` 的 ORT Web 模型。最终模型候选均回到 Rust `composite_inpaint`，不能信任模型自行保留选区外、冻结区和 alpha。

客户端使用固定SHA256的Carve ONNX导出，输入RGB0..1、输出NCHW RGB0..255。约208MB权重首次校验后进入CacheStorage；坏长度、SHA或响应会删除缓存，下一次从已修复文件重新读取。ORT运行在图像Worker：可用时选WebGPU并允许本地WASM算子回退，无GPU使用WASM。COOP/COEP允许最多4个本地WASM线程，部署应发送与Vite相同header；无隔离退回1线程。

正面积RGB/蒙版共用采样footprint，保留比例并对称补至512，冻结/透明RGB不进入已知上下文。模型候选只缓存目标RGB（32MiB LRU），因此12MP原图的小选区不会因完整RGBA超过预算而反复推理。同源模型预览被新参数替代时保留计算/session；旧调用返回AbortError，结果仅交付最新参数。换原图立即取消旧Worker。客户端模块首次加载失败会单独重建Worker以支持重试。

真实3图、缓存与图层回放记录见 [浏览器QA](../qa/client-inpaint/browser-client-review.md)。

## 后端实现

Go 使用 Gin/Viper，配置 `background.worker_url` / `IMG_BACKGROUND_WORKER_URL`。模型推理独立于 Go 进程，Python FastAPI/Torch 负责 `POST /v1/inpaint`，multipart 字段 `image/mask/protect_mask`；返回与展示方向原图同尺寸的 RGBA PNG。参数表单对应 `max_side/context_mode/context_padding/mask_dilation/seam_blend`。

Go 将完整图拆为“Rust 前段 → Rust canonical masks → HTTP LaMa → Rust canonical composite → Rust 后段”，保持图层顺序、几何变换后的尺寸和所有参数校验。禁用与零透明度节点仍走 Rust 校验但不调用模型。模型错误返回 422/503，超时和取消沿用 504/409。

模型候选在公共合成之前缓存。键包含图像字节、公共 target/protection PNG、模型 URL、已固定权重 SHA256、预处理版本和模型设置。仅完整有效 PNG 的成功结果进入 64 MiB LRU；更换蒙版、前段或设置会失效，修改后段调色或非零 opacity 可复用。失败不缓存。

## 原始论文与复现范围

[LaMa，WACV 2022](https://arxiv.org/abs/2109.07161) / [官方代码](https://github.com/advimman/lama)。服务端采用 IOPaint 引用的公开 TorchScript 导出。其 989 个 generator 参数与运行 buffer 全部与官方 README 链接的 `big-lama` checkpoint 逐 tensor 精确相等。证据见 [weight-verification.json](../qa/background-lama/weight-verification.json)。这是原始推理权重的运行包装，未复现训练、数据集指标或后续更强生成模型。

自身包装包括：EXIF 方向统一、蒙版/冻结/透明 RGB 排除、正面积 BOX 缩放与一致蒙版 footprint、模 8 对称 padding、目标约束合成。可选 Poisson 是 [Pérez 等 SIGGRAPH 2003](https://doi.org/10.1145/882262.882269) 的梯度域思想，属于额外接缝处理，不是 LaMa 论文组件。

## 完成标准与测试用例

- 候选真实来自模型推理；不以加亮、传统 blur 或 exemplar 代替。
- 相同图、相同公共蒙版，选择区外/冻结/全部 alpha 逐字节保持。
- 原目标、冻结人物和 alpha=0 隐藏 RGB 不得污染推理输入；缩放不能在已知背景制造黑 halo。
- 前后图层顺序、尺寸、软蒙版、opacity、blend 与普通编辑图一致。
- 客户端模式不能被 Go 自动上传至模型；模型不可用明确报错。
- 输出需有真实照片原图/结果/局部对照，复杂纹理失败必须保留；自然度评分独立于单测。

Python 11 项包装测试：空/冻结目标无模型 identity、选区边界与 alpha、原目标/保护/隐藏 RGB 非局部污染、缩放黑 halo、尺寸/设置、缺失/坏权重、Poisson 常量偏移/线性梯度、真实 FastAPI 路由。Go 测试覆盖实际 Rust 混合 graph、前后翻转顺序、冻结与 alpha、opacity 只合成一次、缓存命中/失效/有界淘汰、不缓存失败、禁用参数检查与客户端导出围栏。

## 当前进度

- [x] 本地 CPU TorchScript 推理与原始权重一致性核对
- [x] Rust 统一蒙版与候选合成，WASM/native 接口
- [x] Go 模型代理、混合导出与成功结果缓存
- [x] 无 LLM 的手工选区调用与失败明确返回
- [x] 包装合同测试、三张双端真实推理与浏览器实际图层回放
- [x] 缩小图像的黑边污染定位、修复及真实街景复测
- [ ] 所有复杂人物/建筑场景自然度达到目标标准
- [ ] 各浏览器客户端性能与大图稳定性验收

## 对账位置与运行

- Rust：`core/image-core/src/model_inpaint.rs`、`src/bin/image-render.rs`；CLI `--inpaint-masks` 输出 source/target/protection PNG 目录，`--candidate` 对单个 LaMa 节点合成。
- Go：`services/api/internal/api/model.go`、`render.go`、`model_test.go`、`cmd/server/main.go`。
- Worker：`services/background-worker/worker.py`、`app.py`、`tests/test_worker.py`。
- 权重安装、真实图和失败证据：[Worker 说明](../../services/background-worker/README.md)、[LaMa QA](../qa/algorithm-audit-lama.md)。

```bash
make client-models       # 可选浏览器模型，安装后可以本地推理
make background-models
make background          # 另一个终端保持运行；默认 loopback 8091
make background-test
make dev                 # Web + Go；已有 API 在代码更新后需重启
```

服务端模型不会在启动时自动联网下载。Python 环境与模型缓存分别位于 `.venv` 与 `.models/background`，不进入 Git。

## 变更记录

- 2026-10-13：接入原 LaMa 权重的独立本地 Worker、公共合成、Go 分段图和缓存；查明 Lanczos RGB/BOX mask footprint 不一致导致的假纹理并修复；保留旧失败证据；增加 client/server 显式执行约束。
