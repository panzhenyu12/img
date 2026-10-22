# 下颌等面积目标曲线：独立 QP 实验

日期：2026-10-22。此项只验证一个新的目标控制器，尚未验证照片审美。**在不减少总内收／外扩面积的前提下，候选消除了声明窗口内原凸曲线的新反向弯曲；位移曲线仍有局部波动，不能据此给出 8 分。**

## 为什么继续修改目标

[连续下颌复核](portable-jaw-continuous-20261022.md)已经区分两个问题：密采样修复了长边内部执行不足，但准确执行连续 beta 目标仍会使原来凸向的左下颌窗口转为反向。因此本实验保持源轮廓和变形场不动，单独调整目标形状。

此前的颈部固定边候选已被后续成片复核拒绝：固定边上端引入了可见台阶。它没有进入本实验，不应把本候选视为“曲线＋颈部”联合改进。

## 输入与控制器定义

输入为用户图原始 63 点轮廓和已有嘴／下巴弧端点。源折线、检测点和语义分区没有平滑、补点猜测或重新标注。左右弧长分别为 143.54472 px、144.21004 px。左弧最后一段源高度下降 0.20553 px，任何触及此段的高度图凸性窗口都被标为未知。

变量是沿原弧长以不超过 2 px 间隔定义的横向位移，区间内线性插值。最终目标折线包含所有原轮廓角点与位移节点。目标为原点位置加人脸横轴乘位移；嘴和下巴端点位移严格为零。

目标函数由按物理弧长积分的 beta 请求偏差，以及固定 4 px 平滑长度的位移二阶差分正则组成。约束包括编辑方向、端点、总有向面积，以及仅在源已凸且高度单调的窗口内不产生新反向弯曲。它是明确的新控制器，未改旧 beta 定义。

面积基准使用当前连续 beta 请求。+100 为 **695.4732327309218 px²**，−100 为其相反数。旧 63 行离散目标面积仅为 627.1380318744923 px²；不能悄悄用这个较小值声称保留当前编辑强度。旧面积只作为另列的对照实验。

## 连续检查与限制

v1 的 0.5 px 约束采样被独立更密查询发现漏检；v2 的加密交换约束仍被独立网格发现边界漏检。这两版保留为失败记录。

v3 在每个源轮廓／位移基函数断点及其 ±16 px 平移处切分窗口中心域。每段中，相邻高度图弦的斜率差分子为二次多项式，分母为正。先在源曲线二次项的根处分区，再在候选各子区间端点与内部二次最小值处检查。原本平直、反向、非单调和弧端 16 px 范围仍为未知。不得把代理折线的直段当作已确认的真实皮肤形状。

连续检查证明的仅是 `[s−16,s]` 与 `[s,s+16]` 两段弦的声明斜率判据。它不证明所有微分曲率、未知区域凸性、成片无缝或自然美观。右侧未知区仍存在反向窗口。

## 结果

两候选的线性可行性检查和 SLSQP 求解均成功，嘴／下巴端点位移为零。连续判据下，左侧 168、右侧 136 个源已凸子区间均未发现超过 1e−8 原始分子数值容差的违例；另用 0.03125 px 独立窗口检查也未发现已知区域新增反转。

| 候选 | 左侧峰位移 | 右侧峰位移 | 总面积变化 | 独立完整多边形面积误差 |
| --- | ---: | ---: | ---: | ---: |
| +100 | 7.88375 px | 8.65211 px | +695.47323 px² | −1.89e−11 px² |
| −100 | 8.84055 px | 8.86625 px | −695.47323 px² | −1.27e−10 px² |

连续 beta 的两侧位移峰约为 8.90722／8.95636 px。候选 +100 减少局部峰值并将面积分配到其他弧段；总面积并未减小。已经打开目标曲线图检查：+100 的位移 profile 仍有局部双峰／凹谷，右侧尤其明显；约束通过不能替代成片判断。

## 导出与复现

实验与生成结果都在忽略目录 `test/results/semantic-deformation-20261022/jaw-equal-area-probe-v3/`，不提交图像、数组或结果数据：

- `analyze.py`：冻结的 QP 与连续检查；SHA256 `b3a3a083879146c93096fbf9de96b3192fd7f800350f288156551b19b8dc280b`。
- `export_profiles.py`：读取已冻结结果并导出，不重跑 QP。
- `plus100-continuous-area-profile.json`：SHA256 `580b17162131369bc41428b9b7efda42cbb63be5484c4b89ce463091cb1fed91`。
- `minus100-continuous-area-profile.json`：SHA256 `fedc3392a6071b4cadbffe4a8a398ff3dd57e272a1748f0e5e1b877807fef0bc`。
- `export-verification.json`：独立插值、原角点、端点与完整多边形面积复核。

两 profile 绑定原 geometry SHA256 `283924cad7a18cd32e7b6c25c8d0824e1115bdb4b09b51fbe046d87ee67c91aa`，编码为 UTF-8 的 `json.dumps(source_geometry, sort_keys=True, separators=(',', ':'))`；同时绑定原 protocol、代码、报告哈希与未知区间。

每侧导出 `{arc_id, arclength_px, across_displacement_px}`。在不变的源弧上求源弧长 `s`，线性插值得到 `du`，目标为 `p(s) + across * du`。数值已经包含 ±100 强度，不得再次乘 amount。弧外仍为原位置。所有原边顶点均被包含，每侧另以 17003 个独立相位查询重建，最大位移误差 1.78e−15 px。

下一步是在同一变形场设置中渲染真实照片，直接比较旧连续 beta 与新 profile；保留眼镜／耳饰／背景保护，但不混入已失败的颈部固定边。若局部位移波动产生可见起伏，应拒绝该曲线候选并研究更合理的源形状不确定性和目标平滑约束，不能放宽测量或压低强度换取通过。

## 可复用模块：v4

已将优化与连续检查抽取到 [jaw_curve_qp.py](../../experiments/semantic-deformation-20261022/jaw_curve_qp.py)，不读取用户图、模型或结果文件。它依赖 NumPy、SciPy，以及同目录 `jaw_collocation.py` 的源几何校验；仍是离线算法实验模块，尚未接 Web／服务端。

```python
solve_jaw_curve_profile(
    geometry, amount,
    max_radius_fraction=.13,
    node_spacing_px=2.,
    smoothing_length_px=4.,
    window_radius_px=16.,
)
evaluate_jaw_curve_profile(geometry, result, arc_id, arclength_px)
evaluate_profile_constraints(geometry, profiles, window_radius_px=16.)
```

第一项返回 geometry hash、面积参考、可执行 profiles 与数值报告。第二项在同一源弧长上输出 source／target／du，并拒绝 geometry hash 不匹配；第三项对任意分段线性 profile 做连续窗口检查，可用于独立查询候选是否违规。求解器、面积、端点、方向或连续检查任一未通过时，第一项返回空 profiles，不导出弱化候选。

默认平滑长度仍为 4 px。可以独立比较 8／12 px 等物理尺度，正则为 `λ⁴∫|d''(s)|²ds`；面积保持硬等式。因此更大的 λ 只能重分配相同的总面积，不能偷偷降低编辑强度。若图像按比例放大，节点间距、平滑长度和窗口半径都需同倍缩放，才能表示同一个几何问题。

v4 明确区别于冻结 v3：连续 beta 在每个原边内是最高 7 次多项式，改用每原边 4 点 Gauss 积分。当前用户 geometry 的 +100 面积为 **695.4732327553754 px²**，比 v3 的密集梯形近似大 **2.44536e−8 px²**；不改变意图，也不宣称逐字复现旧数值。没有修改被冻结的 v3 代码、profiles 或 runner。

[合成测试](../../experiments/semantic-deformation-20261022/test_jaw_curve_qp.py)共 8 项，无任何用户照片依赖。覆盖有解析答案的连续面积、双向编辑与原角点、独立稠密凸性查询、0.02 px 窄缺陷被连续检查发现而 0.5 px 网格漏掉、平直／非单调继续未知、零值恒等与 hash 防错、添加共线源点不变、刚体变换不变、面积不变下的平滑重分配及物理尺度缩放。实际 `jaw_target_profile` adapter 往返测试先复现了抽取时漏掉 `direction`／`original_contour_start_arclength_px` 的错误；修复后覆盖生成结果的 `_validate`、训练与独立探针 `apply_profile_samples`、`profile_contour_targets`，不能只以自身 evaluator 通过为依据。针对该文件运行 `python -m unittest discover -s experiments/semantic-deformation-20261022 -p test_jaw_curve_qp.py -v`，8 项通过，耗时约 1.2 秒。它们验证数学行为，不给真实人脸打分。

## 冻结导出与 8／12 px 候选

[export_jaw_curve_profile.py](../../experiments/semantic-deformation-20261022/export_jaw_curve_profile.py)提供可复现 CLI：

```sh
services/background-worker/.venv/bin/python experiments/semantic-deformation-20261022/export_jaw_curve_profile.py \
  --protocol test/results/semantic-deformation-20261022/user-jaw-continuous-v1/protocol.json \
  --amount 100 --smoothing 8 \
  --output test/results/semantic-deformation-20261022/user-jaw-curve-v4-plus100-smooth8
```

12 px 候选将 `--smoothing` 改为 `12`，并使用独立新目录。已有目录一律拒绝覆盖。导出包括源协议副本、运行的四个本地源码文件副本、参数、Python／NumPy／SciPy 版本、QP 报告与 profile；profile 绑定 source_protocol／qp_source／qp_report／export_source 的 SHA256。运行前后核验源证据与源码未改变。生成后通过实际 adapter 的 `load_profile`，以及训练／探针 727 点和原轮廓接口；另由完整闭合多边形鞋带公式独立复算面积。

[CLI 合成测试](../../experiments/semantic-deformation-20261022/test_export_jaw_curve_profile.py)通过实际子进程导出、冻结源码与协议、adapter 加载、覆盖拒绝和源图身份变化拒绝，共 1 项通过，约 1.1 秒；其“图像”仅为临时哈希身份字节，不解码或依赖任何照片。

已生成的两候选仍为目标曲线，**没有渲染照片、没有添加颈部固定边**。两者总面积同为 **695.4732327553754 px²**，声明已知凸向窗口无超容差违规，adapter 查询误差均为 0：

| 平滑长度 | 左／右峰位移 | 独立闭合面积误差 | 忽略目录 |
| --- | --- | ---: | --- |
| 8 px | 7.61413／7.36877 px | −1.13e−10 px² | `test/results/semantic-deformation-20261022/user-jaw-curve-v4-plus100-smooth8/` |
| 12 px | 7.41716／6.78718 px | −2.00e−10 px² | `test/results/semantic-deformation-20261022/user-jaw-curve-v4-plus100-smooth12/` |

各目录的 `profile.json` 可直接交给成片 runner；SHA256 分别为 `9ec73a3267d2191f5dc9f614f5ef2884bc33e8c53d36761147346c22b50fa86f` 与 `4974847e7a95d30144ff0ed10b859476ff7cbbb1c583520a6519c78ed9254393`。峰值下降是面积重分配；不能单看峰值或平滑指标判断美观，也不能将未约束的源几何区域解释为已经自然。
