# 耳根、耳垂与耳饰附着独立验证（2026-10-15）

本轮未修改生产代码。实验代码和失败候选均保留在 `experiments/ear-accessory-attachment-v1-20261015/`。验证对象是一张用户原图在 0、±30、±100 的重复回放，不把不同像素、左右侧或强度当成独立照片样本，不给出总体审美分数。

## 已确认的问题

1. 耳饰不应在耳垂移动时仍被一律钉死；要求应是耳饰与耳垂的附着关系、形状和材质保持。用户未要求耳朵及耳饰绝对坐标不变。显式用户保护、其他人物和无关背景仍需保护。
2. 整耳平移会拉开上耳根与头部的连接。绕上耳根旋转、耳饰随实际耳垂终点刚性平移，能减少这一项新增缺陷，但没有解决下耳垂硬角。
3. **原 semantic ears 漏掉了真实下耳垂。** 自动源图节点给出的左耳语义 bbox 底部为 y=415，耳饰挂点约 y=431；右耳分别为 y=427、433。左耳 (359,426) 的语义耳覆盖为 0。这些坐标是诊断结果，不是算法内按图片编写的特判。旧 pivot 实际只搬动了耳层里已有的材料，漏掉的耳垂仍被当作 moving skin；数学 `hookGap=0` 不能证明图像里的真实连接点正确。

## 冻结与可复查证据

- 原图：`test/results/user-face-contour-20261014/1024/canonical-source.png`，SHA256 `5f15e92b14c44a468d924af5ac985a7cc7dc1c87d8ab235f2c94b2e88b173c73`。
- 原图 graph SHA256：`9ff494d8434a751833d98fe60f598b179d66813531e0a581013cbbcac33f717c`。
- `export_native_v2.py` 从冻结诊断 binary 导出新 native-v2 的 A/F/background/inverse；四强度最终 RGBA 均与父实验原 native 输出逐字节相同，见 `native-v2-layers/report.json`。
- native-v2 的 inverse 不同于旧 V36；每强度 inverse/source-alpha/F 的哈希单独记录，未以“沿用 V36”代替测量。
- 所有 face field 对照都保留原 inverse，耳部变体没有通过调小全脸幅度掩盖问题。输入干预实验允许 geometry 重新求解，另行记录实际幅度。

## 运动与材料消融

| 实验 | 控制变量 | 观察与判定 |
|---|---|---|
| `outputs/` | 整耳/J直接重建 | 原图颜色与头发丢失，拒绝，保留失败 |
| `outputs-delta-v2/` | 原图残差 + target/source 模型差分 | 解决零运动重建误差，仍有耳根/耳垂连接问题 |
| `outputs-pivot-v3/`、`outputs-pivot-v3-low/` | 同一脸场，整体平移 vs 上根 pivot | 上耳根新增暗缝减少；下耳垂未解决 |
| `native-pivot-v4/` | 精确 native-v2 的 A/F/inverse，fixed / ear-only / pivot-pair | fixed 四组精确等于 native；新 native 附近运动很小，pivot 未带来可接受的下耳垂改善 |
| `native-partition-v5b/` | fixed inverse，局部 alpha-only / color-only / alpha+color | 只改 4–8 像素，漏掉的耳垂不在 semantic ear 范围内；无法解决根因，拒绝 |

`outputs-pivot-v3` 的上耳根暗化代理（相对 fixed 亮度下降 >20 的像素，不是人工标注裂缝面积）：+100 左侧整体平移 185 → pivot 8；-100 为 124 → 0。±30 对应 133 → 2、87 → 0。这只能说明减少了“整耳平移新增的上根暗化”，不能推出“已修复瘦脸”。

native-v4 沿用旧自动纯皮肤 driver，左侧实际 x 位移 +100 为 0.516 px、-100 为 -0.517 px，右侧约 ±0.067 px；旧 V36 同位置约 2.275 / 0.198 px。+30 与 +100 的左局部 driver 几乎相同，说明新 source/attachment 输入改变了局部运动。全脸实际变形需要看父实验 `actual-source-width.json`，不能由此局部值外推。

物理互补对照使用 `E=D*q`、`S=(D-E)/(1-E)`（E=1 时被遮挡 S 置零），保证 `E+(1-E)S=D`。这里 D 是神经联合物理覆盖假设，q 是耳语义归属，二者不是同一种量。`native-partition-v5` 存在误采样背景的问题，脚本已归档为 `history/run_partition-v5-warped-background.py`，该版不用于结论；修正的 v5b 保持背景在目标坐标取样。v5b 的零运动模型误差 ≤1.7e-16、byte 级 identity 保持，写域外和 J≥.95 的金属修改均为 0。

## 新的耳朵神经抠图

`run_ear_matte.py` / `ear-matte-v6/` 对左右耳独立执行 ZIM（原 vendor ONNX），提示全部从源图语义和邻接关系提取：耳内部、下部、direct skin 确认的耳饰邻接皮肤为正点，可靠脸、金属、头发和背景为负点。所有候选保留，使用官方 score argmax，未根据目标修图效果手选。

左右 point 模式官方均选择 index 2，score 为 0.87546 / 0.89952。左邻接皮肤点 (361,429) 的耳 alpha 为 0.98386，右 (658,429) 为 0.99979；负点接近零。真实耳饰挂点本身属于金属，ear alpha 接近零是合理的，不能要求耳和金属在同一像素都为不透明。

实际查看 `ear-matte-v6/{left,right}-point/comparison-1x.png`：新 alpha 补到了原 parser 截掉的下耳垂，耳饰主体仍被排除。这是源材料支持的进展，尚不是完成的瘦脸结果。box 模式全部候选也保留以检查提示稳定性。

## 原生耳输入对照

`run_native_ears.py` / `native-v3-ear-input/` 只把 native-v2 graph 的 `semanticMasks.ears` 替换为左右官方 point alpha 的联合，仍经 BOX 缩放到 ≤512、u8 量化的现有公共合同。其余参数、skin/accessories 输入和 binary 不变。完整渲染会重新求解 geometry，故实际位移必须测量，不能宣称与 native-v2 几何相同。0、±30、±100 的图与层导出保留；运行结果见该目录 report。

| 强度 | 新耳输入最大实际位移 左/右（px） | 同一源坐标测量的平均幅度保留 左/右 |
|---|---|---|
| +30 | 6.006 / 6.006 | 84.63% / 95.35% |
| -30 | 5.817 / 5.638 | 79.07% / 92.34% |
| +100 | 9.271 / 8.491 | 79.93% / 93.43% |
| -100 | 8.257 / 6.903 | 80.60% / 93.59% |

`verify_native_ears.py` 选择 native-v2 实际位移 ≥1px 的源轮廓坐标，分别反解两组 inverse，前向残差 <2.3e-13px。新 mask 改变了原生源轮廓，直接比较 debug JSON 的同行会错配，旧同行比较保留但明确标成无效；正式结果为 `actual-retention-identical-source-probes.json`。最大位移均保留 >80% 并不代表整段曲线保留 >80%：左侧 -30、+100 平均略低于门槛，若干接点甚至被固定到零。

1×实际审图：新输入让耳垂可见形状更完整，但 +100 左耳饰上方仍有细暗楔、-100 右耳下仍有硬角，左太阳穴也需检查新的短纹理变化。因此该输入干预只能证明漏耳垂支持参与了错误运动分区，**拒绝作为当前生产候选**，不能宣称达到 8 分。下一候选应把真实耳材料与可动连接约束联合处理，不能再把新耳整体钉死而压低脸型幅度。

## 回归检查与限制

`test_attachment.py`：金属随耳垂刚性平移、source residual 下零运动 identity、上耳根固定且耳饰保持实际附着向量。三个测试记录了先失败后修复过程。`test_partition.py`：不透明联合皮肤不能因材料分区而出现透明缺口；部分覆盖边缘保留总 alpha。五项检查通过，仅作为算法合同检查，不代替照片审图。

`outputs-zero-check/` 三组零值输出逐字节等于源图。native-v3 的零值也逐字节等于原图。耳/耳饰 alpha 面积与刚性协方差检查只能证明该层采样行为，无法证明原始材料归属正确；本轮发现的漏耳垂正是反例。

## V7：真实材料与旧 native 运动域

`movable-ear-v7/` 使用 neural ear，重新从实际耳饰邻接处选择有 ear≥.95 支持的耳垂点和 direct skin≥.98、ear<.02 的邻近脸部点。旧 native 耳贡献按实际原分区处理：frozen 部分使用 identity，原来漏掉的 lobe 使用 native inverse；先扣除旧贡献再加新 pivot。不能将漏段假定为原先静止后再重复搬动。

本次 driver 位移 +100 左/右为 0.837 / -0.0107px，-100 为 -0.838 / 0.0107px。实际耳垂→金属挂点向量保持，源耳层在耳垂点有 0.98386 / 0.99979 支持，但右侧效果几乎不变，整体仍不可用。这证明旧 native 场在接点的强制抑制不是靠修正素材范围就能解除。

## V8：上耳根约束与分布式接缝失败

克隆 `face-material-fill-v1-20261015/core` 到本实验 `native-root-v8-core`，只增加诊断环境变量 `EAR_ROOT_ANCHORS`。从新源耳材料提取上部内侧接点，以耳高 10% 的圆盘约束上耳根，替代整个耳朵的零速度域；材质/output protection 和其他保护逻辑不改。没有环境变量的控制输出与 native-v3 +100 RGBA 逐字节相同；新 root 模式 0 仍与源图完全一致。冻结 binary SHA256 为 `e460dae8b74b748fec82d5fa5755623bea256f31335d21bd1374ef9f83e5e790`。

`native-root-v8/` 释放下耳后的接点实际 x 位移 +100 左/右为 12.996 / -13.103px；-100 为 -9.242 / 8.406px。`movable-ear-v8/` 据此驱动真实 ear lobe pivot 与刚性 J，挂点相对向量误差为 0，上根位移为 0。全脸同源坐标场幅度保持：相对 native-v2，±100 左右平均位移为 1.63–1.87 倍，没有探针低于 80%；采样 fold=0，最小 Jacobian 为 +100 0.704、-100 0.357。这些仅为场的数值性质。

**实际审图明确拒绝 V8。** +100 仍有大黑楔，旧耳的细轮廓、上耳根裂边、J 旧位置残影；-100 也有耳/J 边缘残留。脸场与耳刚性旋转各自平滑，不代表它们的共同边界相同。源图残差静止叠加在 ~13px 的材料位移下也暴露出旧位置纹理，不能继续当作最终原生 compositor。

`audit_v8_seam.py` / `v8-seam-audit/` 从源 ear alpha 内侧边界取左 83、右 82 个真实界面点，对比同一点的 native face 前向终点与 ear pivot 终点；反解残差 <1.7e-13px：

| 强度/侧 | 当前整条接缝 RMS / 最大误差（px） | 同一上根下最优刚性旋转 RMS（px） | 改成最优刚性后耳垂目标偏移（px） |
|---|---|---|---|
| +100 左 | 3.581 / 4.759 | 2.457 | 4.299 |
| +100 右 | 3.452 / 4.796 | 2.208 | 4.639 |
| -100 左 | 2.191 / 3.004 | 1.467 | 2.726 |
| -100 右 | 1.879 / 2.622 | 1.056 | 2.742 |

最优旋转也不能同时满足整条界面和真实耳垂目标，瓶颈不只是旋转角度选错。停止 V9 及继续像素/遮罩微调。下一方向是共同接口约束下的联合场与共享源材料重采样，设计见 `doc/research/ear-shared-interface-design-20261015.md`。

增加 `observed_lobe_contact` 与 `test_contact.py`：缺失的耳材料不能靠一个数学挂点宣称已附着；真实耳垂必须有源 ear/skin 支持、排除金属，并在挂点采样不确定度内。两个测试先失败后实现，总计 7 项通过，日志 `history/contact-{red,green}.log`。这提供可复用的“拒绝伪附着”合同，不表示 V8 图像通过。

附件下未观测区域使用附近可见背景作为 underlay 假设，尚无真实隐藏纹理 ground truth。此次仅一张照片，不能推广为所有耳型、耳饰、侧脸均已解决；达到“8分”的验收仍需要多图、1×视觉和完整编辑交互。

方法流程参考：Kassis et al., *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*，2026，[doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065)。该来源仅支持假设、消融与证据边界的记录流程，不是美颜算法效果证据；原记录保存于本实验 `history/scientific-agent-skills-arxiv-record.html`。
