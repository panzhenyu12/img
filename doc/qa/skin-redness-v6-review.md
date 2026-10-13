# 局部去红 v6 实拍复核

日期：2026-10-14。本轮改善的是 `skin.retouch/redness<0` 的色度调整：从全选区统一减红，改为优先中和邻域异常红色色度，并恢复同一区域的暖色色度。当前保守暂评 **7.8/10**；它能减弱红斑，但暗色痘印、囊肿凸起和凹陷纹理仍可见，不能记作自动祛痘或整套皮肤功能达到 8 分。

## 同参样片与版本

九张独立照片 × `redness=0/-40/-100/+40`，`smooth=0`，共 36 项。六张人像包括深肤色男性、微笑男性/女性、头发遮挡女性、带妆女性和眼镜街拍；另外三张是明显痤疮局部。旋转派生 portrait-roll 不计独立原照。各照片使用固定输入、同一 graph 的 mask/landmarks/region，不重新检测或改动选区。

基线是 `.run/skin-chroma-v3-image-render`，SHA256 `8aacc21470c2262fb9dfbe1f716bd1f30197fcfcc19c1cb8e569df8cd00751db`。最终 v6 是独立 target 构建后复制的不可变 `.run/skin-redness-v6-image-render`，SHA256 `db142152ab238fe72ff2501d5fc8fc05f191dc58c15fb5ce4081924aa87c6005`；每次 spawn 前后校验 SHA，避免并行构建覆盖共享 binary。完整记录见 [36项报告](skin-redness-v6-final.json)，第一版单红色轴 v5 也保留在 [失败迭代](skin-redness-v5.json)，后者未进入最终生产效果报告。

试跑全部支持域外/alpha变化为 0。负档每像素最大亮度变化不超过 0.50/255，属于最终 RGB 8bit 量化；没有提亮、磨平或更改 alpha 来伪造去红。零值严格原图，正值红润输出与 v3 相同。

六张常规人像 graph 中除眼镜街拍外仍使用旧语义缓存，独立 accessories/ears 缺口保留，不能借一次眼镜照片外推所有配饰。此回放依赖本地忽略的解析缓存；报告记录输入和 graph 哈希，缓存的重新识别结果可能不同，不能把它宣传成任意清洁检出的独立算法验收。

## 算法与回归

以 192px 有界分析栅格统计亮度归一化的红绿色度，在 eligible skin 邻域取 35% 分位；明确去红时放宽参考邻域，避免较大的红斑把自己当作正常肤色。正常邻域的绿蓝色度另作归一化局部估计，避免只减红导致暗红点呈紫灰。去红仍保留少量全局响应，但大部分修正作用于红色异常；最大减幅不允许翻转红绿关系，最终在零亮度色度平面重建 RGB。

donor 继续排除未选、透明、冻结、五官、头发和明确语义保护。两个新增 Rust 回归分别验证“去红优先改善已知红斑、保留正常皮肤、毛孔和亮度”与“冻结/semantic protection 的红或青色不能污染邻域参考”。宽斑第一版回归实际失败后扩大参考邻域；v5 实拍仍出现暗点偏紫灰，v6 才增加第二色度恢复。没有修改图片、标注或通过门限掩盖失败。

此前已运行 color release 测试 55 项通过；最后集成源再次以 `color::tests::redness_` 运行两个新增回归，均通过。默认不请求去红的回放见 [默认输出对账](skin-redness-default-parity.json)：36 项 `redness=0`（其中 18 项自然/强/纹理磨皮）与 v3 PNG 逐字节相同；6 项 `redness=+40` 也完全相同。只有 6 项明确的负值去红变化。因此磨皮默认输出没有因本次肤色算法回归。

## 实际效果

全部 gentle/strong 的整体扫描见 [温和档](skin-redness-v6/gentle-contact.png)、[强档](skin-redness-v6/strong-contact.png)。常规照片同时用 100% 支持域裁切检查脸、头发、眼睛、唇齿和脖颈；痤疮图用固定 100% 局部检查。

| 样片 | 目视结果 | 证据 |
|---|---|---|
| 微笑男性 | 旧全脸强去红有灰绿倾向，v6 保留主要暖肤色和鼻/颊自然色；毛孔不变 | [100%](skin-redness-v6/fixture-portrait-strong-comparison.png) |
| 深肤色男性 | 减轻鼻部/颊部过红，避免把整个深肤色统一变灰；阴影与头发保留 | [100%](skin-redness-v6/portrait-dark-strong-comparison.png) |
| 头发遮挡女性 | 比旧强档保留更多健康暖色；额颊高光及发丝纹理保留 | [100%](skin-redness-v6/portrait-hair-strong-comparison.png) |
| 带妆女性 | 健康肤色变化较轻，旧强档颈部整体灰绿减轻；口红与眼妆固定 | [100%](skin-redness-v6/portrait-makeup-strong-comparison.png) |
| 微笑女性 | 全脸灰绿减轻，颊部局部色差变淡；原有纹理/笑纹仍可见 | [100%](skin-redness-v6/portrait-woman-strong-comparison.png) |
| 眼镜街拍 | 肤色整体较自然；眼镜、饰物与街景目视固定，本图原瑕疵少 | [100%](skin-redness-v6/portrait-glasses-street-strong-comparison.png) |
| teen | 红斑颜色显著减弱，强档暗痘印仍可见，不能算完整修瑕 | [100%](skin-redness-v6/acne-teen-strong-comparison.png) |
| oily | 红丘疹色度降低，油脂反光与白头的亮度形状保留；不是白头修复 | [100%](skin-redness-v6/acne-oily-strong-comparison.png) |
| cystic | 红色减弱但囊肿暗色/凸起仍明显；选区外负例固定 | [100%](skin-redness-v6/acne-cystic-strong-comparison.png) |

标注处的相对红色色度（中心/固定环形皮肤中值差），`redness=-100`：teen 1 从 10.856→-0.022，teen 2 从 7.698→0.632；oily 从 8.063→0.231；cystic 从 7.882→3.686。该数值只描述局部去红，不自动换算自然度。v6 保住亮度，所以无法修复暗斑的明暗结构，也没有把这些残留当作成功。

按效果35%、自然度/细节35%、部位保护20%、效率10%暂评：效果7.7、自然度8.2、保护8.0、效率6.5，总计7.845，记 **7.8**。保护项仍因旧配饰缓存和姿态覆盖不足保守；效率仅观察 native 总进程，最终18项负档约 124–637ms，中位数259ms，包含 PNG I/O/协议，受同时编译与其他回放负载影响。没有浏览器热帧、峰值内存或低性能设备证据，不按实时优化达标给高分。

下一步针对明确局部瑕疵做颜色、明暗和纹理的联合修复；补独立侧脸、胡须、低光、混合照明与配饰照片。白头 v4c 重审的高光误判/环状残边已确认并撤回，见 [失败报告](skin-bright-rejected/review.md)。
