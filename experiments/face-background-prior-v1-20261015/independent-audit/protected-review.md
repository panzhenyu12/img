# 软耳/饰品保护修补的独立复核

针对旧 mixed-edge union 中 38 个 soft ears/accessories 像素被改的问题，作者生成 `mixed-edge-protected/`，在目标域明确排除 `ears/accessories >= .05`。本复核用独立 SciPy 双线性重建前景/alpha，验证保存输出。

结果：observed-only、virtual-context 均满足 output alpha 不变、B 前景/alpha重建 max byte error=0、头发核心0变化、目标域外0变化、soft ears/accessories 新变化0。目标像素由 union 15,515 降到 15,477，排除38像素；initial B 与 source 的既有耳饰差异仍42像素，单列继承，不冒充新修复。

复现：

```sh
services/background-worker/.venv/bin/python experiments/face-background-prior-v1-20261015/independent-audit/check_mixed_edge.py mixed-edge-protected
```

该修补只补写域合同，不改变 B 层序、源 alpha、前景或几何。上下文 RGB 和 inference mask 仍同时变化；3/3 隐藏条带测试显示 virtual 纹理误差较全 unknown 更差，故没有画质通过结论。原失败 mixed-edge 仍保留。
