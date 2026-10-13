# 算法整改与验证记录

日期：2026-10-13。状态：基础性质验证已建立，真实照片的全部自然度验收尚未通过。世界领先质量是目标，当前没有支持该结论的实验或盲评证据。

## 撤销的旧判据

旧逐参数测试只检查“像素发生变化”。真实人像发现，旧默认瘦脸脸内变化为0，脸外变化199,063像素；轮廓bbox的衰减和固定源polygon二次裁切使脸缘不动。这不是合格的瘦脸。旧116项像素变化及其截图只保留历史联调证据，不代表当前算法验收。

当前已接入参数由FUNC-010与operation-catalog自动生成。未完成专用算法的脸高、面中、双下巴、眼袋/泪沟结构、部分鼻部结构、身高腿长、体态与衣物版型均隐藏，核心拒绝非零调用。手工液化仍可使用。身体参数与部位不匹配、缺少关键点/subjectMask时明确报错。

## 实现与效果合同

| 功能 | 修正后的算法/测试 | 真实照片边界 |
|---|---|---|
| 裁剪/缩放/旋转/翻转/透视 | 坐标、维度、identity、透明边角；public graph黑盒 | 8-bit RGBA，不含RAW/ICC |
| 网格与液化 | 像素距离；所有mode方向、有限支持、归零、冻结；连续stroke在取样前参与保护 | 手工二维编辑，不是三维语义重建 |
| 九个人脸轮廓参数 | Schaefer rigid MLS、多尺度F/B颜色估计、内脸/头发/耳朵/饰物独立层；双向真实轮廓、滚转、五官/第二张脸/ROI/alpha保护 | 外扩细色线、发际线受冻结限制、下巴暗点与颈部阴影仍未通过 |
| 五官与对称 | 像素脸轴和实际semantic polygon；逐参数正负方向与目标面积；鼻子矫正依据真实偏斜；对称保留各侧纹理 | 侧脸/遮挡/缺少可靠语义不可冒充可编辑 |
| 腰/肩/胯/上臂/前臂/腿 | 必需真实subjectMask、65截面双侧轮廓、单调逆映射、F/B分层与完整soft fringe、partMask接口；缺mask旧背景warp已删除 | 真实15项全部未达8；自动Pose无真partMask，手/发/包/持物隔离及窄带仍有失败 |
| 皮肤与妆容 | 有蒙版Fast Guided Filter、512px分析、共享亮度guide、连续高频残留；skin、feature、冻结控制取样；肤色明度/色度与阴影高光保护；唇/牙/眉独立边界 | 真实48项皮肤及123项五官/肤色批测保护检查完成，自然度仍需逐项复核；不含学习式瑕疵/妆容重建 |
| 衣物 | 褶皱细纹弱化、纹理/色相/饱和度/亮度；强度连续，取样隔离 | 严重折痕、服装版型重建未实现；[衣物多图](clothing-multisample-review.md) |
| 去除杂物/人物 | 保留传统exemplar；浏览器ONNX和服务端TorchScript LaMa真实推理、共用Rust target/composite、面积预处理、冻结/alpha、图层顺序与有界缓存 | 简单街景改善，复杂轨道和建筑结构仍失败；浏览器首次效率未达标 |
| 仿制/修复/绘色 | 明确source坐标、均匀背景瑕疵修复、alpha与冻结 | 传统手工工具 |
| 调色/曲线/HSL/LUT/胶片 | 线性光EV、渐进黑白、不同空间尺度、保边降噪、DCP去雾；LUT三线性与channel顺序，滤镜强度0完全原图 | 胶片为效果模拟，没有富士官方配方一致性证据 |

测试位置：`core/image-core/src/{face,body,color,geometry,inpaint,mask,engine,validation}.rs` 与 `core/image-core/tests/algorithm_contracts.rs`。颜色逐项证据见 [颜色审计](algorithm-audit-color.md)；去人证据见 [补全审计](algorithm-audit-inpaint.md)。

## 当前工程检查

最终Rust release全量检查：141项库测试+10项public graph黑盒测试通过；2项ignored分别为真实街景手动检查和native人脸性能基准，不能计入默认通过数量。`clippy --all-targets -- -D warnings`、全部Go test/vet、TypeScript/Vite生产构建通过；Python模型包装11项通过。模型安装脚本校验face parser与LaMa权重。

去雾曾在native/WASM产生1字节差异，现统一软件libm并使用weighted f64 box guided filter；正负强度与分析缩放/冻结三场景严格RGBA 0差异，不放宽容差。具体证据见颜色审计。最新完整Playwright数量、真实检测与模型性能见 [最终检查](current-validation.md)。浏览器真实LaMa完整图层回放无API请求，与独立Rust合成严格一致；不能将模型执行成功当作画质验收。

## 多张真实照片

`test/images/manifest.json` 保存来源、下载地址、尺寸、SHA256和标签；覆盖肤色、眼镜、胡须、长发、歪头、侧脸、身体、裙装、衣物纹理、多人街景与建筑背景。派生滚转照片注明来源。

`make quality-face` / `make quality-body` 在本地模型和Rust WASM中生成原图、输出、放大差分、操作图与摘要。原始操作含大raster，保存到忽略的`test/results/`；`doc/qa/`保留精简指标和代表对比图。报告保留FAIL、UNVERIFIED、检测缺失和不可用部位，不把重检测噪声当作真实轮廓测量，不把任何像素变化当成功。

当前版本见 [人脸目视](face-contour-visual-review.md)、[身体V2](body-contour-v2-review.md)、[五官与皮肤](feature-skin-multisample-review.md)、[LaMa双端](algorithm-audit-lama.md)、[浏览器LaMa](client-inpaint/browser-client-review.md)。旧 [人像批量](portrait-multisample-review.md)、[身形批量](body-multisample-review.md) 保留历史版本失败，不能混为当前算法的逐项验证。真实照片效果与工程单测分开记录。

## 可复现检查

```sh
make setup
make check
make stop
make dev
make e2e
make quality-face
make quality-body
```

真实街景的ignored单测需要fixture环境变量，重现方式见补全文档。它不会被普通`cargo test`静默算作通过。

## 后续质量门槛

先改善脸缘matting、skin/hair与脖子/衣物底层、身体遮挡与服装像素解析、补全接缝，逐案例验证。再增加多肤色/姿态/光照/分辨率样片、量化轮廓与背景指标、真人盲评和成熟工具对照。工程链路通过不是这些自然度门槛完成的证据。
