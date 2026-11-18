# 公共图入口眼宽成片独立复核（2026-11-18）

## 结论

user、dark 两张来源的 `eyeWidth=±30/±60` 八张成片中，眼角按正负值真实收放，虹膜外观未被横向拉扁；原尺寸未发现新增眼角裂口、重复眼睑线或眉毛形状损坏。**这两张输入及已测四档的眼宽响应、可见虹膜保持和所查接点可接受。**

这不覆盖眼睑开合、眼睛高度、眼镜/头发遮挡、侧脸、大幅极值或多控叠加，也不等同五官整体已可用/8 分。user 来源有“AI 创作”水印，不将两张统一称为相机实拍。本次只独立读取成片和源几何，没有修改 core、重检测或构建。

## 实际入口与审查范围

- 回放脚本：`experiments/eye-material-public-20261118/replay.py`。
- 公共入口：`core/image-core/target/release/image-render --input … --graph … --output …`，通过公共 rlib 解析图并执行，不使用 test-only material hook。
- 二进制 SHA256：`0d6476cb70b5eb0b6686f4ce31bdd14f4a0c924de21581c60ab5e19fbdf99212`。
- 冻结源观察：`test/results/eye-height-layers-20261117/raw-source-observations/{user,dark}`；回放先核验 source 与 observations SHA，再取同源 MediaPipe 眼裂、虹膜中心与四边界点。
- 输入 `eyeGeometry.leftEye/rightEye.iris` 与原眼裂 polygon 均来自这些源观察；没有使用输出图反推、手工 QA 探针或生成式目标。
- 输出：`test/results/eye-material-public-20261118/{user,dark}/{-60,-30,+0,+30,+60}.png`。

独立看过两张源全图、全部八张非零全图、各自 `eye-native.png` 与 `eye-2x.png`，并针对 +60 内眼角/鼻根连接额外做原图对照。用户图眼角外扩/内收可见，dark 也有相同方向响应，不以 changedPixels 非零替代这一观察。零值与原 RGBA 相同由回放报告记录；本次重点是实际非零成片。

## 虹膜及非目标区复核

按公共 graph 的源 iris center/四点形成投影椭圆，与源 aperture polygon 相交，以像素中心决定内部集合。直接比较源/输出 RGBA，而非只看 feature center：

| 来源 | 左/右可见虹膜像素数 | ±30/±60 内部变化 |
|---|---:|---:|
| user | 280 / 292 | 0 / 0 |
| dark | 253 / 245 | 0 / 0 |

这些是**源测量所定义的可见虹膜交集**，不是像素级人工真值标注，也不把被眼睑遮挡的预测虹膜弧线当成可见材质。数值保持与成片中虹膜尺寸、高光和纹理外观不被横向拉扁相互支持，但不能单独证明全部眼形自然。

两个来源全部八张输出的眉毛、嘴唇、牙齿语义 polygon 内部变化均为 0。变化范围围绕眼部：user 最宽的 +60 范围为 `x292..468,y255..298`；dark 为 `x263..451,y423..459`。全图未见耳饰、发缘、脸颈和背景被带动。

## +60 的保护限界

鼻 polygon 不是四档都逐像素保持：

| 来源 | +60 鼻 polygon 内变化像素 | 最大单通道差 | 位置 |
|---|---:|---:|---|
| user | 17 | 21 | 约 `x368..392,y276..282`，内眼角旁鼻根侧缘 |
| dark | 20 | 23 | 约 `x345..368,y436..444`，内眼角旁鼻根侧缘 |

±30 和 -60 的鼻 polygon 内变化均为 0。+60 的变化紧邻实际移动的内眼角；独立局部对照没有发现新的鼻形坏影或接缝，当前观察更符合眼角过渡支持与鼻根语义区域的重叠。故不把它列成已证实不可用缺陷，但**不能宣称本轮鼻部全像素不变**；如果产品合同要求鼻 polygon 完全冻结，需要单独处理冲突并保留眼角响应，不能只截断输出造成新接缝。

局部对照在各来源子目录 `nose-root-intersection-4x.png`，顺序 source/+60，nearest 放大仅用于定位。完整原尺寸检查仍是主要画质依据。

## 复核数据

`test/results/eye-material-public-20261118/report.json` 保存公共入口结果、source/observations/graph/output SHA256。新增独立测量仅在忽略目录 `independent-material-audit.json`，包含逐档输出指纹、变化包围盒和语义交集保持结果，不把图像、大坐标文件或模型加入仓库。

后续不能用本批通过替代对 eyeHeight/eyelid 的材料分离验证；这次解决的是有同源虹膜测量时的**眼宽与虹膜保持**，没有生成被眼睑遮住的新纹理。
