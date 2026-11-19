# 当前普通入口的大腿、小腿与上臂回放

2026-11-19。`apply_graph_json` → 普通 `body.adjust`，复用既有检测 graph，固定 `0/−30/+30` 三档。**fashion 可见左大腿、yoga 左小腿和裸露左上臂在这两张既有照片的轻档有限可用**。实现者、独立 reviewer 和 root 均查看实际图片；本轮没有修改这些部位的算法，不代表所有身形功能达到 8 分。

## 输入与执行

| 样片 / 操作 | 原图 SHA-256 | 原 graph SHA-256 |
|---|---|---|
| fashion-standing / left-thigh | `16ec09161b769d5d2553a03ab271dd30a737a88165885384b654ea74b2731cf1` | `710d3bb21a844ff8501e49c1a2c4376fceb6b4b5bf1e459c3dfe96ca95b6cba2` |
| body-yoga-mediapipe / left-calf | `7a648e911e8cda8fb54bc6f2d12f3aa8c57d05719121607bd31e8dec6958cc11` | `6a369659decc7703df4eed3d635efe2a9ad8060f3f2a6409c390f09e1c570a5d` |

只改变对应的 `adjustments.thighs` / `adjustments.calves` 请求；原部位、骨轴、选区、subject/part/protect mask 不变。未重跑模型、未使用预制源分层、未构建 Web/Go/WASM。仅定向构建普通 Rust CLI：

```sh
cargo build --release --manifest-path core/image-core/Cargo.toml --bin image-render
```

本次构建 16.35 秒。renderer SHA-256 为 `4ef7f7313fbda54a845737be6912c8d527e357aec05634004e681512073d2323`；`body.rs` SHA-256 为 `78d405df630fdf85b74a0e0f2ebd9ee41931daa4356b76dad71c1c6f8d8e1a51`。逐档 graph 哈希、执行结果和图片留在忽略目录：

- `test/results/body-limb-diagnosis-20261119/fashion-standing-left-thigh/ordinary/report.json`
- `test/results/body-limb-diagnosis-20261119/yoga-left-calf/report.json`

## 实际结果与范围

两组零请求均与源图逐 RGBA 相等。源保护 raster 在全图像素中心独立采样核对：fashion 的 24,831 个保护像素、yoga 的 9,365 个保护像素，三档变化均为零；画布 alpha 也不变。

fashion 大腿 −30/+30 分别改变 7,505/7,128 个像素。原尺寸全图和 3× 腰带、口袋、会阴至画面下缘的裁图显示可见外缘双向宽窄变化，没有复现旧报告中的硬裂口、离体衣条或突兀断接。原裤缘折线有所变化；**膝和踝在画外，不能据此验收其衔接**。原骨轴中点处同件裤料与另一腿连通，固定双边 RGB 扫描未闭合，测量如实记 `null`，没有换窗口追求数值通过。

yoga 小腿 −30/+30 分别改变 2,420/2,150 个像素。原骨轴中点法线的固定 RGB 色差扫描，不读取候选 alpha/位移场：源可见宽 27.055 px，−30 为 29.004 px（+1.949），+30 为 25.157 px（−1.898）。这是一条截面的响应代理，不能代替审美评分。原尺寸三档全图和 4× 膝下至踝部对照显示连接连续，未见新增双边或台阶；源边颗粒在原图已有。

独立 reviewer `/root/global_review_1119` 完成上述图片检查，root 复核同意。结论限定在既有照片、指定可见部位和 ±30，不外推交叉肢体、遮挡、强档或所有腿部照片。

## 一次不合格的源层复用

此前尝试复用 fashion 的腰部 prepared arrays。数组虽为全画布，腿部却只有静止占位，运动区域的 rear alpha 全零：腿场实际非零且最大位移约 4.466 px，输出仍为原图。该结果已标记 `inadmissible-reused-regional-source-layers`，**不能当作瘦腿无效或瘦腿通过的证据**。上层目录的这组 layered 输出不参与本次验收；有效结果均来自 `ordinary/`。

这说明预制层的数组尺寸和源重构通过，并不证明材料支持覆盖了另一个编辑部位。已验收的腰部源层只用于已审范围，本轮没有为此扩展接口或修改源层。

## 重放

以下命令从仓库根目录运行。逐档 graph 已冻结在本地忽略目录，原图路径及其哈希见各自报告；将 `+30` 换为 `-30` 可重放反向。fashion 的零档 graph 位于其 `ordinary/+0.json`，yoga 的零档位于该 case 的 `+0.json`。

```sh
core/image-core/target/release/image-render --input test/results/body-parts-v14/fashion-standing/source.png --graph test/results/body-limb-diagnosis-20261119/fashion-standing-left-thigh/+30.json --output /tmp/fashion-left-thigh-plus30.png
core/image-core/target/release/image-render --input test/results/body-anatomical-inputs/body-yoga-mediapipe/source.png --graph test/results/body-limb-diagnosis-20261119/yoga-left-calf/+30.json --output /tmp/yoga-left-calf-plus30.png
```

照片、数组、二进制与生成结果均不纳入 Git。

## 同一 yoga 照片的单侧上臂补充

随后仅增加 `left-upper-arm` 的 `0/−30/+30`，使用同一普通 renderer，不再构建或运行模型。原 graph SHA-256 为 `b7eb4ae49075e6204ac14619b7a25c90b50c0d0d9ee45af8d5bebe89a0bdce26`；证据位于 `test/results/body-limb-diagnosis-20261119/yoga-left-upper-arm/`。零请求逐 RGBA 相等，9,365 个源保护像素和画布 alpha 均不变。固定源骨轴中点的 RGB 截面宽度为 23.284 px，−30 为 25.103 px（+1.819），+30 为 21.580 px（−1.703）；该数值只证明一条截面的真实响应。

独立 reviewer 查看源图、±30 全幅及 5× 肩至肘对照，确认轻度肌肉宽窄双向可见，肩、腋窝、肘衔接没有新增硬裂缝或台阶；天空与皮肤边的软颗粒在源图已有，未确认新增离体轮廓。结论为该既有照片单侧裸露上臂 ±30 有限可用，不外推衣袖、遮挡或强档。无新增算法补丁，不继续围绕已验收样片调参。
