# 混色背景补全 soft ears/accessories 保护复核

`mixed-edge-protected/` 排除了旧 target∪B-write 中38个 soft ears/accessories 像素。独立 SciPy 重建确认两组候选的前景/alpha与保存输出逐字节一致、目标域外和 alpha 无变化、耳饰新增变化0。initial B 相对源图已有42像素差异单独记录。

这是写域安全性修补，不是前景遮挡、几何或背景纹理质量证明。virtual-context 仍与 observed-only 同时改变上下文和推理 mask，隐藏条带测试不通过，不上线。

证据：[protected-review.md](../../experiments/face-background-prior-v1-20261015/independent-audit/protected-review.md)。
