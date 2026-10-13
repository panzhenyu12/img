# 当前版本验证与已知失败

日期：2026-10-13。本页记录算法整改后的版本，旧phase1-validation为历史联调记录。

## 工程检查

| 检查 | 实际结果 |
|---|---|
| Rust release全量 | 119库测试+10黑盒测试通过，1真实街景测试ignored（另行手动验证） |
| Rust clippy all-targets -D warnings | 通过 |
| native/WASM release | 编译完成，同一Rust graph核心 |
| Go test/vet | 全部通过，包括22类native HTTP输入与5项未定位/未实现参数拒绝 |
| TypeScript/Vite生产构建 | 通过 |
| 本地模型 | MediaPipe已有；faceparser SHA256验证通过 |
| 去雾跨端回归 | 正60/负60、271×139分析缩放与冻结；RGBA逐字节差异0 |
| 完整Playwright | 24/24通过（独占运行约4.6分钟），所有公开滑杆native/WASM逐字节差异0 |

## 真实照片结果

- 人脸批量14样本：9张检测成功，41个参数案例中40个输出、1个明确补图拒绝；5张未检测到。成功输出支持范围外变化0、四个五官中心保护。5个faceWidth案例仍有固定背景保护FAIL；其他参数像素检查通过也不代表自然度验收。
- 人脸native目视7图：6图真实脸缘内收，1图侧脸未检测。faceWidth路径下颌原白色/黄色误补背景线消除；耳饰、近脸发丝和细边缘仍未达专业自然度。makeup/roll的强chin参数仍有耳下黄色三角填补，不能扩展该修复结论到全部轮廓参数。
- 身形15案例：14个输出固定背景与支持域外变化0；1个右上臂因无合法底层donor拒绝。部分腰/腿在重检测测量中无法证实收缩；裙下遮挡腿部隐藏。不能用输出数当作自然度通过数。
- 衣物3图21案例：选区外变化0；细纹弱化和颜色变化成立，大折痕/版型仍保留。
- 去人街景：选中人物已移除、保护及选区外不变，release约1.5–1.9秒，放大仍有块状接缝。

原图、结果、差分、图形参数和全部失败见 [人脸批量](portrait-multisample-review.md)、[人脸目视](face-contour-visual-review.md)、[身形批量](body-multisample-review.md)、[衣物批量](clothing-multisample-review.md)、[去人审计](algorithm-audit-inpaint.md)。大raster操作图保存在忽略的test/results，源照片与精简报告/代表图入Git。

## 重现

```sh
make setup
make check
make dev
make e2e
make quality
```

make quality依次运行全部多人脸、身形、衣物诊断，前一个失败仍运行后续；有失败时最终返回非零。当前真实质量报告未全部通过，不能以工程单测或浏览器回归代替专业/世界领先自然度验收。CPU/Canvas2D、8-bit RGBA，尚无GPU大图或多浏览器性能承诺；自动检测和解析需要本地模型，手工工具独立于AI。

## 性能实测限制

在本机独占运行中，真实人脸、Pose、肤色/妆容、裁剪后检测的浏览器完整案例约42.4、48.1、54.7、31.5秒，包含模型检测、渲染及自动化操作，不是单次算法耗时。这是CPU Demo链路，不能宣称实时或大图高性能。测试ready等待使用明确60秒预算，单test仍90秒，像素一致性和效果断言未放宽；第一轮与多人像批测争用CPU曾发生超时，独占复跑通过。
