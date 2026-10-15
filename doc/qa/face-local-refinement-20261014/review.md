# 瘦脸露底的无材质条件 LaMa 消融

日期：2026-10-14。结论：**不接入生产**。局部补全可以组织为异步层，但这个直接修复已合成图像的方案会把部分露底带补回脸皮、削弱内收，并保留细边。不是几何/合成已经正确的完整混合管线，也没有测试近期的条件扩散模型。

输入是未验收的 [face-side-v12 候选](../face-side-v12/review.md) 原照 +100 输出，1024×1024。输入 SHA、debug alpha/mask/geometry SHA、LaMa checkpoint SHA 和四个输出 SHA 见 [report.json](report.json)。本实验只有一张原始输入、一个强度，不能推断其他照片或整个神经补全类别。

目标来自记录的物理覆盖率：`old > 0 && new <= 1/255 && holes && !frozen`，共 5,875 个像素。[target.png](target.png) 和 [frozen.png](frozen.png) 已保存。为防止掩盖形变错误，此实验不重绘可见新主体边缘；半覆盖边缘也未进行完整的 underlayer 重合成。因此模型写入范围安全，但并不具备真正的层条件修复。

| 配置 | 模型输入 | 推理 ms | 总 ms | outside / frozen / alpha 改变 |
| --- | --- | ---: | ---: | --- |
| crop64 | 416×320，无缩小 | 3,151.86 | 6,902.22，含首次加载 | 0 / 0 / 0 |
| crop128 | 512×424，已缩小 | 6,871.75 | 6,957.41 | 0 / 0 / 0 |
| crop128 + Poisson | 512×424，已缩小 | 5,095.59 | 5,245.88 | 0 / 0 / 0 |
| full | 512×512，原图缩小 | 3,761.64 | 3,856.94 | 0 / 0 / 0 |

上述是本次 CPU/3 线程、不同冷/热状态且可能与其他测试竞争的单次耗时，不作速度排行榜，不能外推 WebGPU。至少没有支持当前工程“100–200ms”的性能承诺。Poisson 求解收敛也不代表图像正确。

![从左到右：未验收候选、crop64、crop128、crop128+Poisson、full；原尺寸下颌区](jaw-comparison.png)

面部、耳饰等冻结输出仍可作为模型的已知上下文，冻结合同由目标排除和最终合成保证。不能直接复用背景去人模式中“保护人物也排除为模型 donor”的语义；否则连需要的脸颈上下文也会丢失。

下一步需从连续形变产生正确的新轮廓与 coverage，区分应露出的脖子/头发/场景，再让模型只预测对应隐藏层，最后以物理 alpha 合成。不能因模型返回了锐利纹理就接受重新长宽的脸。

重放：使用新的输出目录运行 `services/background-worker/.venv/bin/python scripts/review_face_disocclusion.py --candidate test/results/user-face-contour-20261014/1024/trimap-candidate/faceWidth100.png --debug test/results/user-face-contour-20261014/1024/trimap-candidate/debug100 --output test/results/user-face-contour-20261014/local-refinement-rerun`。原 debug 数据和完整输出在忽略目录 `test/results`；来源代码和未接受几何 patch 保存在 face-side-v12 文档目录。
