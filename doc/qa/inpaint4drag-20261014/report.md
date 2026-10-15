# Inpaint4Drag 六档隔离实验：不接入生产美颜核心

核验日期：2026-10-14。本次实际运行官方 ICCV 2025 Inpaint4Drag 的双向像素 warp，六档为 `0、+30、−30、+60、−60、+100`。结论为 **REJECTED**：可借鉴“显式形变 → 局部补全”的分层接口，拒绝把未修改的官方 warp 加现有 LaMa 作为脸型编辑核心。高目标 mask IoU、冻结合同通过，仍不能证明自然轮廓或五官保持。

## 范围与可追溯输入

官方固定提交 `bfb0fafd456ac01ee23ade0cc04b64c9781eaa87`，未修改 `utils/drag.py`；[源码](https://github.com/Visual-AI/Inpaint4Drag/blob/bfb0fafd456ac01ee23ade0cc04b64c9781eaa87/utils/drag.py)、[论文](https://arxiv.org/abs/2509.04582)。没有导入会自动加载 SD/LCM 的官方 UI。本实验的补全器是本项目已有 LaMa，**没有运行论文的 SD1.5 + TAESD + LCM**，故此结论只覆盖官方 warp 与已测试补全组合，不是对论文全部模型的复现评判。

同一张用户原照 canonical 1024²：SHA256 `5f15e92b14c44a468d924af5ac985a7cc7dc1c87d8ab235f2c94b2e88b173c73`。检测只在原照运行，使用归档 graph，SHA256 `9ff494d8434a751833d98fe60f598b179d66813531e0a581013cbbcac33f717c`；没有重检测修改图来制造成功。各档实际目标由冻结 native candidate 分别导出；不是简单线性放大一个目标。`geometry.json` 点坐标为原图绝对坐标，`origin` 仅表示 debug mask 裁剪原点。

官方整数点对按 UI round，浮点目标另存。native SHA256 `235f3337e1ba453d4721bd427efa59c66e26a027847c0d86227a6613a9612621`；官方 warp SHA256 `6d209b11461f52f201125e944e1cda7e2ce3728580969210c46de7bb70e70e92`；执行 runner SHA256 `372098cdbe2110f64cb403e79f6178107b2a7c8a37d8362f7634e1d3110f78c9`。LaMa 权重 SHA256 `344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9`。更多输出哈希见 [experiment-report.json](experiment-report.json)、[review.json](review.json)。

CPU，Torch 2.2.2、NumPy 1.26.4、OpenCV 4.8.1，3 线程；OpenCV 单独安装至 `.run/inpaint4drag-deps`，未改变后台服务 `.venv`。这些是本机时间，不能引用论文 V100 的 0.01s/0.3s 作为本机或浏览器指标。

## 实测与失败证据

| 档位 | 官方 CPU warp ms | warp 目标 mask IoU | 裸输出发/耳/饰品改动像素 | 裸输出眼区改动像素 | 鼻区改动像素 | 唇区改动像素 | LaMa inference ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 独立 identity 核验，未计时 | 同一控制点 | 0 | 0 | 0 | 0 | 跳过 |
| +30 | 222.75 | 0.999346 | 0 | 0 | 3,341 | 2,652 | 5,108.38 |
| −30 | 155.79 | 0.999532 | 13 | 0 | 1,607 | 2,482 | 6,119.50 |
| +60 | 137.36 | 0.998800 | 0 | 58 | 3,652 | 3,118 | 3,531.74 |
| −60 | 145.85 | 0.998909 | 13 | 0 | 1,843 | 2,782 | 3,625.64 |
| +100 | 146.35 | 0.997966 | 0 | 167 | 3,630 | 3,323 | 3,729.61 |

鼻/唇/眼是原照解剖多边形内部的像素变化监测，边缘按 round 光栅化；不是 Rust feather coverage、位移误差或修改图重检测。像素变动数量本身不是身份分数，但对照图确认 +100 鼻口缩小、−60 鼻口加宽，存在明显非目标联动。

主 runner 的 0 档只验证 native identity，其 `official_identity_pixels:0` 是静态占位，**不能作为官方 warp 已运行的证据**。随后 `review.py` 独立调用未修改官方 `bi_warp`，同源/目标整数控制点，实际 RGBA 变化为 0；但官方仍产生 **8,027 像素**的轮廓补全带。因此 0 参数必须在调用层跳过补全，否则 identity 会被补全模型破坏。

对 +100 的右轮廓原生像素做独立探针：`y=480` 的目标 mask 右端为 `x=616`；`(630,480)` 在目标外、补全允许区内。原图 RGBA `[160,115,94,255]`，native `[16,15,11,255]`，裸官方 warp 保留 `[160,115,94,255]`，warp+LaMa `[158,115,90,255]`；`(624,480)` 的 LaMa 也为 `[161,114,98,255]`。这说明旧脸皮肤残留与 LaMa 再生使自然边缘仍靠近旧轮廓。**移动 mask 准确不等于最终皮肤边缘移动准确**。

视觉复核还发现：裸 warp 有双轮廓/旧皮肤残留；LaMa 消掉黑缝却重新长出脸宽；正负档耳根附近可见白缝。逐像素冻结外围只能实现修改权限，不能单独保证与被冻结耳朵/头发的几何连接自然。

![+30 右轮廓对照](faceWidth30-right-comparison-2x.png)

![−60 下颌与鼻口对照](faceWidth-60-jaw-comparison-2x.png)

![+100 与既有局部补全对照](faceWidth100-previous-lama-comparison-2x.png)

## 合同及其边界

附加合成层把原照头发/耳朵/饰品 coverage≥0.5 和 source/target 并集 +5px 以外的原像素复制回来；所有非零档保护版 warp/LaMa 的这些区域变化及 RGBA alpha 变化均为 0。眼睛只监测，未靠冻结眼睛掩盖 collateral。扩张必须允许目标外轮廓写入，所以没有要求“原脸之外所有背景永远不变”这种矛盾合同。

官方 warp 使用 top-4 逆距离位移、整数映射、最多 100 个逆映射 reference；`RETR_EXTERNAL` 加实心轮廓会吞掉内部洞。未提供连续场 Jacobian、无折叠或背景保直保证。本次不把整数 donor 差分当成无折叠证明。

## 采用与拒绝

- 采用研究层的可替换接口：原图、形变 mask、source/target 点对 → warp、露底、边界补全 mask；明确补全权限、冻结后合成和逐档验收。
- 拒绝原样 warp + LaMa 进入生产瘦脸；不以 ≥0.997 mask IoU 或消掉黑线接受效果。
- 继续独立检查可动主体 alpha、无折叠局部几何场和局部补全条件。更好 source alpha 只能修复源边界/权限，不会消除旧场 `alpha=1→0` 或内外位移连接问题。
- 只有一张独立源照片，五个非零参数档不是五张照片。没有人脸 identity embedding、ground-truth alpha、跨照片统计或浏览器移植验收；不能声称 8/10、世界一流或整篇论文复现成功。

## 复现

```sh
uv pip install --python services/background-worker/.venv/bin/python --target .run/inpaint4drag-deps --no-deps opencv-python-headless==4.8.1.78
services/background-worker/.venv/bin/python experiments/inpaint4drag-20261014/run.py --lama --out test/results/inpaint4drag-new-run
```

原始结果在 `test/results/inpaint4drag-20261014/`；`review.py` 专门读取该目录并创建本归档，已有归档拒绝覆盖。需要全新评审时先改为新输出/归档路径，不能覆盖已签名证据。实验源码见 `experiments/inpaint4drag-20261014/`。
