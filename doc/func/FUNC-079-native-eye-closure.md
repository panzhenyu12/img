# FUNC-079 普通 Rust 眼部轻闭合

2026-11-19。状态：**普通 Rust 原图入口已完成；用户原图和 makeup 第二轮人工材料修正通过限定轻档审图。自动识别仍有失败，不是整套功能达到 8 分。**

本功能把原图、完整源关键点和可选分类标注转为一次准备的眼部源层，再用整数 `[-30,0]` 生成轻闭合结果。它消除历史 instrumented test binary、预制层数组和旧位移场输入依赖。语言迁移本身不等于画质提升；源材料估计仍需承担既有的纹理误判和隐藏底色不确定性。

## 范围与接口

这是独立的普通 native API，不开放通用图操作 `feature.adjust` 中待完成的 `eyeHeight` / `eyelid`，也不代表 Web、WASM、Go 或 MCP 已接入。

| 入口 | 合同 |
|---|---|
| `prepare_eye_closure(source, normalized_landmarks, materials)` | 原始解码图像、至少 478 个完整 MediaPipe 归一化源点、按眼命名的可选材料标注；拥有源图并只准备一次。 |
| `PreparedEyeEdit::reports()` / `geometries()` | 返回源层报告及源定位结果，供独立审查；定位曲线不是 alpha。 |
| `PreparedEyeEdit::render(amount)` | 复用同一源准备，接受整数 `-30..=0`，返回渲染图像；不重新检测或根据编辑结果修源层。 |
| `PreparedEyeEdit::render_diagnostic(amount)` | 一次计算返回图像及该档位实际使用的流场，供迁移核对。 |
| `eye_closure_params_from_landmarks(width,height,points,amount)` | 转换源坐标并产生限定参数；不增加新的点标注要求。 |
| `prepare_eye_closure_field(source,params)` | 输出绝对源 `pixel-index` 反采样 XY、完整 stationary mask 和积分步数；不调用测试入口或读取磁盘场。 |

`amount` 是既有眼高形变驱动的轻档参数，**不是眼裂关闭百分比**。默认回放仍为 `0/-15/-30`；同一源准备的 31 个整数档已有数值扫描，−5/−22 另作实图审查；不将整数参数说成已验收的连续实数滑杆。拒绝正值、低于 −30 的值、小数及额外五官调整参数。

本期范围是有完整源观测的不透明人像轻闭合。自动睁眼、重新生成完全隐藏眼球、眼镜遮挡、明显侧脸、浓密交叠睫毛、任意五官组合和多脸同时编辑不在通用可用承诺内。多脸输入只能显式选择一张脸，不能静默取第一张。

## 源定位与坐标

`eye_localize.rs` 的源定位以同一解码 RGB 和原始关键点为输入：建立各眼眼角坐标系，以有序上下睑点形成 PCHIP 解剖先验；只有源图中可确认的侧眼白对比证据参与有限局部修正。使用同一源亮度/色度噪声、源采样和曲线约束，不读取目标形状或旧样本拟合文件。两张原图同解码的源曲线最大差小于 4e-11 px；关键点或拟合曲线仍不是光学混合真值。

原始 MediaPipe 点表示归一化物理图像位置。进入 native 像素索引时只做一次转换：

`native_normalized = (physical_normalized * dimension - 0.5) / dimension`。

流场内部不再减 `.5`。源曲线的原点/采样也使用一致的整数像素索引合同。分类标注本来就是原图 `pixel-index`，整数 XY 表示像素中心，不能再移半像素。

人脸 `region` 是原物理归一化 FACE_OVAL 外边界加 20% padding，不是特征点；保持该边界不移位。超出合法像素中心域的必需观测明确拒绝，不裁剪关键点来继续运行。

形变保留原轻闭合实验的单眼眼角轴、紧支撑速度场、源眉毛保护及已有积分方式。不会把眼球和眼睑当作一张纹理一起压扁；最终效果取决于后续材料合成是否正确。

## 源材料与合成

分类标注沿用 [FUNC-078](FUNC-078-eye-source-material-correction.md)，由原文件 SHA-256、宽高及 `pixel-index` 坐标绑定。每只眼允许矩形、多边形和离散圆笔划；标签冲突、跨眼归属冲突、源绑定错误或非法域均拒绝。输入不包含真实 alpha、隐藏眼球底图或输出位移场。

| 标签 | 行为 |
|---|---|
| `ocularAnchors` | 明确源眼球观测，`B=C`、`E=1`；可用于源底色供体，并纠正自动假前景。 |
| `movingLashSupport` | 明确细睫毛及其与同动眼睑相接的混合边；不是纯黑睫毛颜色或 alpha。 |
| `unknown` | 身份或有效源支撑未解决。编辑若依赖此处或其实际反采样足迹则拒绝；零值保持需单独验证。 |

普通源估计保留 V1 的源法向底色、源皮肤/暗色 proposal 及有界三色混合思路。明确移动材料使用指定眼球供体的局部稳健表面：固定 Huber IRLS、源残差 MAD 尺度及单侧拟合不确定性；前景允许源皮肤和睫毛颜色混合。

有限、有界、具备源供体并能重建源 RGB 的层才可使用。对已由分类标注确认身份的材料，经验颜色模型近似误差单列 `modelUncertain`，不重新称为身份未知；缺供体、域外或数值非法仍阻断。此许可不授予自动检测的可疑暗线，也不证明估计 alpha 或隐藏颜色是真值。

合成采用 `C'(x)=R(q(x))+E(q(x))*B(x)`：同动前景只随眼睑场采样一次，眼球纹理保留原位置。源层复用到全部强度；不靠最后裁切 RGB 掩盖越界或拼接问题。

## 普通 CLI

实现：`core/image-core/examples/eye-closure.rs`。以下为普通原图入口的可复现命令。在仓库根目录运行；测试照片与材料仅保留本地，输出目录必须尚不存在。

```bash
cargo run --manifest-path core/image-core/Cargo.toml --release --example eye-closure -- \
  --source test/results/user-face-contour-20261014/768/canonical-source.png \
  --observations test/results/eye-height-layers-20261117/raw-source-observations/user/observations.json \
  --materials test/results/eye-closing-source-20261119/manual-materials-user/materials.json \
  --output-dir test/results/example-native-eye-closure-user-001 \
  --amounts="0,-15,-30" \
  --save-fields
```

`--materials` 在接口上可省略，但无标注 V1 已有高光误判，不应作为自动可用路径推荐；其作用是修正源材料身份。`--face-index N` 用于显式选择多脸输入。原图 bytes 用同一次读取完成 SHA 计算与 Rust 解码；拒绝透明输入，以及 EXIF orientation 需要旋转/镜像的原图，不静默改变观测对应关系。

观测 JSON 使用现有 `dimensions / rows[id=full] / box / faceCount / sourceLandmarks` 格式。宽高、完整图区域、面数和选中索引均校验。相邻 `report.json` 存在时必须同时匹配原图与观测 SHA；确实没有绑定报告时，只有显式添加 `--allow-unbound-observations` 才允许继续。该选项不能绕过已存在但错误的绑定。

材料 JSON 保持既有 `version=1.0 / source{sha256,width,height,coordinates} / eyes` 格式。中性的 `feather=0`、`invert=false` 在校验后兼容处理，非零 feather、反转、连续 alpha 或任意未声明字段不接受。

产物为同一 Rust 解码的 `source.png`、`candidate-zero.png`、`candidate-minusN.png`、输入观测/材料副本及 `report.json`。报告记录实际 `strengths`、源准备次数 1、源几何与材料报告、每档时间、成功或拒绝原因。每档发生失败时不伪造结果图。

`--save-fields` 只输出实际使用的诊断场：`candidate-*.inverse.f32` 为小端绝对源 XY，`candidate-*.stationary.u8` 为完整逐像素 mask；报告记录路径和 SHA。它不是场输入选项，默认关闭以避免大数组占用。

已完成原三档迁移对照和新增整数扫档；实图结论以明确标注版本为准。例如同一源准备扫描全部整数，不保存大场数组：

```bash
cargo run --manifest-path core/image-core/Cargo.toml --release --example eye-closure -- \
  --source test/results/user-face-contour-20261014/768/canonical-source.png \
  --observations test/results/eye-height-layers-20261117/raw-source-observations/user/observations.json \
  --materials test/results/eye-closing-source-20261119/manual-materials-user/materials.json \
  --output-dir test/results/example-native-eye-closure-user-sweep-001 \
  --amounts="0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30"
```

只审新增两档时可用 `--amounts="-5,-22"` 并换一个未存在的输出目录。CLI 拒绝重复强度，避免同名产物被覆盖。

## 迁移对照与验收门槛

旧样片不是同一自动估计器，必须保留版本差异：

| 证据 | 源估计基线 | 不能据此推断的结论 |
|---|---|---|
| 历史 user 人工材料回放 | `source_material:estimate_legacy_source`，即 V1 | 不能用该图的通过覆盖自动 V5/V6/V7 的失败。 |
| 历史 makeup 人工睫毛回放 | `observed_ocular_v5:estimate_observed_ocular` | 不能要求新 V1 路径逐像素等于旧 V5，也不能把版本差异称为 Rust 迁移 bug。 |
| 本次 native 迁移 | 普通源估计 V1 思路 + 显式材料修正 | 不因换语言而宣称清晰度、自然度或自动识别升级。 |

公平迁移核对使用 native 输出的 `source.png` 作为双方同一解码源，在原始观测上重新估计源曲线，并复用本次 native 实际场；Python 参考使用对应 V1 与同一分类标注。不得读取历史层数组作 native 输入，或将 Pillow 与 Rust 的 JPEG 解码差异计作算法误差。源曲线/裁切、源层、流场及最终 RGB 的差异应分别定位。

| 验收项目 | 完成标准 | 当前状态 |
|---|---|---|
| 普通 native 构建与入口 | 不依赖 `cfg(test)`、环境观察器、旧二进制或预制场 | 见下方实际记录 |
| 0/−15/−30 迁移 | 同源解码、同场、同版本参考，源准备一次；零值准确保持；差异有解释 | 见下方实际记录 |
| 源材料行为 | 眼球纠正、睫毛一次运动、未知依赖拒绝、源绑定与冲突检查，沿用独立最终 RGB/位置门槛 | 见下方实际记录 |
| 31 档数值扫描 | 有限场、采样网格正面积、眉毛固定、眼内整体真实运动；不要求每档每像素都改变 | 见下方实际记录 |
| 新增 −5/−22 与实图 | 轮廓闭合有效；无新增裂缝、双睫线、旧位残块、硬标注边和可见高光损伤 | 见下方实际记录 |
| 范围与默认系统边界 | 通用 `feature.adjust` 不解禁；不把实验推广为睁眼/眼镜/侧脸通用 | 见下方实际记录 |

相关代码：`eye_localize.rs`、`eye_source.rs`、`eye_material.rs`、`eye_closure_field.rs` 与 native `eye-closure` example。本次实际结果如下；[FUNC-078](FUNC-078-eye-source-material-correction.md) 的 Python 实验通过记录不能代替这次 Rust 迁移验收。

## 实际验收（2026-11-19）

新增 15 项定向行为测试已通过：明确材料 4、源定位 3、场/档位与精确优化 5、公共源入口 3。`eye_` 定向筛选合计 46 通过、1 个已有照片依赖测试跳过，另公共源入口 3 项通过，后续精确支持域优化 1 项独立通过，源残差缓存变更后公共源入口 3 项再次通过；这些数量不是画质评分。

普通 CLI 从原图、478 点观测和冻结标签重新准备 user/makeup 的三档；同解码、同场的新 Python V1 参考与 Rust 在 `0/−15/−30` 最终 RGB 逐字节一致。它验证了迁移没有新误差，也暴露了 V1 本身的缺点：makeup 原有标签只修睫毛，未标的右眼高光 `(695,370)` 仍从 `[235,240,255]` 错变为 `[81,92,121]`。该首轮完整双眼不通过，不能用旧 V5 的结果遮盖。

随后明确进行人工第二轮修正：根据原图坐标网格增加两个可见眼球 polygon，保留原 26 个睫毛 moving 像素；首个左眼上边 6 像素超几何域的版本在出图前被拒，显式将该标注上边从 y=367 改为 y=368，几何不动。最终材料 JSON SHA-256 为 `8a8f6b181f85ab4e6836258a1b5c49b58b29e5b9747579e682de2e9684966fee`，在重新出图前冻结。这是交互式人工纠错，不能记作盲测、自动识别改善或新算法评分。

该次 `0/−5/−15/−22/−30` 同解码同场参考仍逐字节相同。独立审图覆盖两只完整眼、−30 全脸、源标注及 10× 虹膜/标注接界：右侧失真高光恢复，四个非零档均保持原 RGB；未见新增硬标注边、暗环、双睫毛或眼睑断线。原 26 个 moving 像素仍可估计，13 个 `modelUncertain` 保留。用户原图采用既有 61/68 像素 ocular polygon，轻档独立复核有限通过，左眼小残差仍保留。

两张原图的首轮同一 prepared 对象各扫 31 档，源准备只一次；有限/正采样 cell/静止像素保持及真实响应检查通过。user 最小采样 Jacobian 0.92947、最大相邻档位移 0.02121 px；makeup 对应 0.93048、0.04132 px。这是几何与采样检查，不把首轮 makeup 的视觉失败改成通过。第二轮人工标签随后在优化后的相同普通入口也完成全31档，仍无采样翻折或未知依赖，最小Jacobian/相邻档变化保持相同；其中已审五档成片与优化前哈希一致。完整输出只保留选定档位，审计核验 SHA 后清理本轮重复场和中间图约 616 MB。

实图与独立结论见 [独立审查](../qa/eye-rust-material-route-20261119.md)。本地证据 `test/results/eye-closing-source-20261119/native-v1/` 下保留 user、首轮 makeup、makeup-manual-r2、原图标注及 sweep-audit；这些照片、标注和场均不推送。

## 保持成片的计算优化

源残差 `R=C−E×B` 也在 prepare 阶段计算一次并复用原 RGB 浮点缓冲；每次调节不再分配并分解整幅照片。眼部场只在确切两椭圆支持内进入通用求值；匹配原速度函数的 f32 运算顺序。原积分器已有零初速提前返回，所以该优化只减少支持外覆盖读取与分派，不声称原来额外执行了 32/62 次积分。

新增水平及分别倾斜/亚像素偏移两组眼睛 × 全 31 档、全像素逐 f32 bit 对照，优化与原映射完全相同。真实 user 与 makeup 第二轮修正的五档输出、field、stationary SHA-256 均与优化前一致，未重新调整标签或形变精度。

同一机器、固定 makeup 1280×1128 原图与第二轮标签、五档顺序运行三次，−30 渲染中位数约 `310→282 ms`，−5 约 `223→180 ms`，零值约 `82→41 ms`。源准备中位数约 `293→333 ms`；源分解前移一次换取后续滑杆复用，不隐去首次准备成本。计时不含 PNG 编码和模型检测，机器非独占，不能泛化为浏览器性能或宣称 60fps。

## 变更记录

- 2026-11-19：登记普通 native 源准备、参数化轻闭合、源绑定 CLI 与迁移验收合同；完成原图迁移与人工第二轮纠错的有限验收，保留首轮高光失败及自动材料识别边界。
