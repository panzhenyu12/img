# ZIM 组合提示接入与原照片消融

状态：隔离实验，修正了模型调用语义；不等于完整瘦脸通过，未接生产。固定 ICCV 2025 ZIM 官方模型和 vendor，源码文件保持原样。

## 发现与修改

上游 `modeling/decoder.py` 的 `points` 分支生成点坐标/标签后，`boxes` 分支再次赋值，覆盖为两个框角及 `[2,3]`。与此同时 `predictor.py` 在有点时用所有点坐标生成 attention；它不区分正负标签。所以旧 `box+positive+negative` 调用实际是“框 token + 所有点的 attention”，不能解释成模型接收了负点却不遵守它。

`adapter.py` 显式将点 token 与框角拼接为 `[1,0,...,2,3]`，有框时不添加仅用于无框提示的 padding。分别保留原点 attention 和框 attention，避免把两种改变混为一个因素。只有框/只有点时仍调用未修改上游。当前只验证单图 batch=1，不声称通用批量支持。

`run.py` 用三个冻结原照、两组自动提示和五种模式实跑 30 次，保存全部 120 张 float alpha：旧 mixed、box-only、points-only、combined-point、combined-box。模型权重和源码 hash 见 `outputs/report.json`。六次旧 mixed 与此前原始 alpha **逐浮点完全相同**；不存在换图、重检测或挑选图片后比较的问题。

## 实际观测

以下比较同一自动负点协议下的官方最高分输出，数值只是与冻结语义核心的相容性，**不是物理 alpha 准确率**。

| 原照 | 旧 mixed 可信 skin 平均 alpha | 正确 combined-box | hair 核心平均 alpha（新） | 负点 alpha（新） |
| --- | ---: | ---: | ---: | ---: |
| portrait-woman | 0.75835 | 0.01016 | 0.88576 | 1.39e-7 |
| user-face-contour | 0.01709 | 0.01293 | 0.96523 | 4.94e-8 |
| portrait-hair | 0.77273 | 0.00201 | 0.97008 | 2.86e-8 |

修正后的 portrait-hair 实际排除了原先被大面积选入的脸部；box-only 在这张图也很好，因此不能把全部收益归给“更复杂的提示”。portrait-woman 单正点本来已较好，修正调用并不能消除肤色/发丝混色边缘，也不解决背景补全或遮挡前后关系。`points-only` 单正点在 woman 仍选入大量脸部，失败输出保留。

2026-10-14 最早的 `experiments/zim-20261014/run.py` 采用 box 与 points 互斥分支，**不受本次组合覆盖问题影响**；其负点/耳朵失败不能一并撤销。受影响的是后来同时传 box 和 point 的接入协议及其因果解释。

## 回归与复现

`test_adapter.py` 三项合同直接检查送入 ONNX 的真实 feed、attention 因素和单提示委托，全部通过。独立真实 ONNX 动态 token 与覆盖复现见 `../zim-semantic-holdout-20261015/` 的审计记录。模型和图像质量仍须额外原照复核，不能因 token 合同通过给予 8 分。

```sh
services/background-worker/.venv/bin/python experiments/zim-combined-prompts-20261015/test_adapter.py
services/background-worker/.venv/bin/python experiments/zim-combined-prompts-20261015/run.py --out /absolute/new/output/directory
```

默认输出目录必须不存在，防止覆盖既有成功和失败证据。原输出在 `outputs/`，新推理必须指定新的目录。
