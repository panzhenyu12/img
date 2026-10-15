# 用户耳饰：固定提示的五种 ZIM 调用对照

隔离实验，2026-10-15。仅使用已有 1024×1024 原始 canonical RGB 与检测 graph；不改生产、不重新检测输出、不变形、不改变任何 raw alpha。图像含 AI 创作水印，不声称已核实为相机实拍。

`run.py` 从 ZIM-v2 的 `prompts.json` 逐字读取两个 accessories 组件，并重新生成同一自动提示验证相等。每个组件只使用已有 box 与最深内部正点，没有负点，不新增手工区域。一次 encoder、十次 decoder、每次四个原始候选。

`adapter-snapshot.py` 是本轮冻结的 corrected mixed adapter。`upstream-mixed` 和 `box-only` 实际 labels 为 `[2,3]`；`points-only` 是 `[1,-1]`；`combined-point` / `combined-box` 是 `[1,2,3]`，分别沿用 point / box attention。只验证 B=1。原上游 mixed 会覆盖 point token，不能称为模型忽略了正/负点。

```sh
services/background-worker/.venv/bin/python experiments/zim-accessory-combined-20261015/run.py
services/background-worker/.venv/bin/python experiments/zim-accessory-combined-20261015/audit.py
```

`run.py` 拒绝覆盖已有 `outputs`。复现实验应在新的同级实验目录复制冻结脚本与 adapter；保持所有输入不变。`audit.py` 可复算当前原始输出的诊断，所有阈值来自既有 ZIM-v2 ownership 审计。

- `outputs/report.json`：模型/source/graph/脚本 hash、实际 ORT tokens 与 attention hash、官方分数、时间。
- `outputs/component-N-MODE/alphas.npy`：完整 `[4,1024,1024]` f32 raw alpha；不量化、不修边。
- `outputs/component-N-MODE/component-candidates-{1x,4x}.png`：全部四候选与原图/parser；4x 为 nearest 放大，仅辅助检查。
- `outputs/audit-report.json`：全部四十候选的排除区、源皮肤证据、未知区、连通孔洞与源高亮诊断。
- `outputs/component-N-audit/`：全部五模式官方 argmax 的对照、源支持区与 unknown 区。用于下一步 F/B 实验，不是可直接替换的生产 mask。

详见 [评审](review.md)。
