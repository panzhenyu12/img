# v18 颈部连接约束：一次候选，拒绝合入

从 v17 最终修复出发，只有源颈部连接约束改变。此候选使用短连续可见 neck 侧边缘，在眼坐标系拟合局部直线，并与源 faceContour 求交。不硬编码用户照片坐标，缺少连续 neck 证据时不创建连接。连接成为真实固定控制；同一 attachment 距离用于目标可行 taper、QP、速度、shooting，同一 inverse flow 运输 physical alpha。

用户 canonical 原照检测到连接 `(410.03,542.70)` / `(612.55,533.32)`。这仍是基于可见语义与 person 支持的外推，没有 amodal 真值或三维几何，不等价于证明隐藏颈部材质。

六档 ±30/±60/±100 原图均已保留。ears/accessories/hair RGB 变化0，9个实际五官内部0，alpha0；纯收缩scene背景0。6档 stride2 / 112320 个有限差分探针无 fold；最小Jac .557–.894。这些只检查硬约束与数值，不构成视觉验收。

**拒绝：**大黑楔明显缩小，但颈部仍有硬阴影、皮肤拼接与侧连接小黑点。更关键的是固定neck连接与现有无新增凹曲线约束让范围内目标失去可达性，实际瘦脸强度显著下降。不能称为保留请求幅度，不能给 8/10，不能合入生产。

| faceWidth | 请求最大位移px | 可行最大位移px | 实际源polygon最大位移px |
|---:|---:|---:|---:|
| +100 | 20.020 | 8.202 | 8.152 |
| -100 | 20.020 | 12.521 | 12.702 |
| +60 | 12.012 | 6.790 | 6.308 |
| -60 | 12.012 | 7.950 | 7.951 |
| +30 | 6.006 | 3.830 | 3.824 |
| -30 | 6.006 | 4.123 | 3.973 |

最后一列为同一速度场对原63点polygon的实际积分，不是QP控制点请求，也不是最终 face+neck 合成轮廓真值。控制点与polygon位置不同，故 -100 的实际最大值可略高于控制点的可行最大值；它仍低于请求20px。没有降低用户滑杆值，但约束本身实质压低了效果。

![+100 因果对比](faceWidth100-causal-jaw-2x.png)

![−100 因果对比](faceWidth-100-causal-jaw-2x.png)

![六档 full face](six-strengths-face-montage.png)

本轮按上限只做这一个 neck attachment 候选，不再迭代阈值或后处理。后续若继续，需要同一结构内的可滑动连接或 face/neck joint 物理边界，并重新验证实际位移、真实RGB连接和多原图。不能仅固定点后宣称全幅瘦脸通过。

独立 target 最终完整 face 单测：**37 passed / 3 failed / 1 ignored**。三个失败未放宽：rolled 0.3rad 只能推到一侧连接；jaw_reveals_neck 测试中实际轮廓未收缩；uncertain_face_neck 测试中旧阴影 `(54,186)` 仍为源 RGB。除视觉/幅度拒绝之外，这些也是明确的功能回归。日志见 `independent-unit-test.log`。

源码、二进制、全六档原输出/graph/diagnostic SHA 保留。共用 Cargo target 曾错误复用旧 v17 单测程序（没有新neck测试），该日志无效并保留；独立 target 重编译日志才用于最终结论。生产 `core/image-core/src/face.rs` 与公共WASM未改。

独立 target 从最终归档源码重建 native renderer 后，六档 PNG 与原冻结候选再次全部 byte exact；见 `independent-rebuild-output-parity.json`。这验证源码/测试/渲染证据对应，仍不改变视觉拒绝结论。
