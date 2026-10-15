# ZIM 自动选择的独立原照与接口验证

三张独立原照的 50 组五模式对照已完成，真实 feeds 均通过合同核对；尚未达到物理 matte 或修图视觉验收。当前任务是验证自动选择和点框联合调用能否用于后续 source physical matte。生产代码、vendor、模型权重和旧证据不变；所有 alpha 只是候选，原 parser 只是语义参考，并无 matting 真值。

## 已完成的因果证据

- `raw/`、`selection/`：makeup/glasses 原照的旧 mixed 调用。6 组、每组四原始 alpha，保留官方 argmax 与固定旧门槛自动选择。
- `old-protocol-review.md`：旧调用实际没有传递点标签的纠正，以及通过门槛仍存在整脸软 alpha 幽影的负例。
- `decoder-contract.py/json`：捕获旧 wrapper 在固定坐标翻转正负标签后，所有实际输入完全相同；真实 ONNX 可接受 4 和 7 个动态点/框角 token，输出 finite。
- `decoder-label-flip.py/json`：正确拼接 token 时，仅负点标签 0→1 即改变所有 1,048,576 个输出 logits，max Δ=31.113、mean Δ=4.497，score max Δ=.1444。使用合成零特征，仅验证接口因果，不是图片效果。
- `dark-detection/1920/`：第三张独立原照 dark 的新原图检测，原始尺寸 1280×1920，不替换旧 graph，不把 resize 派生图计为新照片。

## 正确联合调用的独立对照

`combined.py` 使用 root 的独立 adapter，原 vendor 不变。三张 holdout 原照均跑 upstream-mixed、box-only、points-only、combined-point、combined-box；自动正/负点请求协议不调参。每个模式仍保留全部四个候选，不按最终效果挑图。

加载前强制固定 encoder/decoder SHA；ORT recording proxy 原样转发输入，仅记录实际点坐标、标签、attention SHA/shape/sum。所有模式的 feed 都核对，没有假定请求即到达模型。只验证 prompt batch B=1；不声称通用批处理。

`combined-raw/report-with-feeds.json` 保存50组实际输入。`combined-selection/report.json` 保留固定门槛选择及所有候选的软alpha/耳/饰品/皮肤核心审计，`combined-summary.json` 为简表，`combined-review.md` 记录结论。每张原图仅 `set_image` 一次，全部协议复用该图编码特征。root runner/adapter 的使用快照在 `combined-code-snapshot`，当前执行期间代码 hash 必须保持不变。

## 复现

输出目录必须不存在，不允许覆盖失败证据。模型和依赖复用已有本地固定缓存。

```sh
services/background-worker/.venv/bin/python experiments/zim-semantic-holdout-20261015/run.py
services/background-worker/.venv/bin/python experiments/zim-semantic-holdout-20261015/audit.py
services/background-worker/.venv/bin/python experiments/zim-semantic-holdout-20261015/decoder-contract.py
services/background-worker/.venv/bin/python experiments/zim-semantic-holdout-20261015/decoder-label-flip.py
services/background-worker/.venv/bin/python experiments/zim-semantic-holdout-20261015/combined.py
```

makeup 使用全尺寸冻结 source/graph；glasses 使用原 JPEG 1280×853，原检测 graph 来自此前 768×512 的同原图，归一化坐标映射到完整原照，不缩小 ZIM 输出。检测置信图的有限分辨率仍是限制。dark 的完整原图 source-only graph 单独存档。
