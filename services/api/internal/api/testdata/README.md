# HTTP 图像操作测试输入

`portrait-localization.json` 与 `portrait-subject-mask.png` 对应共享照片 `apps/web/tests/fixtures/portrait.jpg`。

- JSON 保存归一化的人脸外轮廓、五官 polygon 和 MediaPipe 轮廓 landmarks；本次从 `doc/qa/native-face-contour/fixture-portrait-graph.json` 捕获到独立测试输入，运行测试时不依赖 QA 输出路径。
- PNG 无损保存该照片的 410×512 人物 alpha，测试转换成公共 `subjectMask` 的 raster 数据；原图为 820×1024，公共 mask 以归一化坐标匹配不同尺寸。
- `face.adjust` 在 HTTP 测试中使用显式 polygon，`feature.adjust` 使用五官定位，`body.adjust` 使用人物 matte、指定上臂区域和手工上臂轴。衣物只测试已实现的选区内颜色调整。
- 这些数据用于检查 HTTP → Go → native Rust 的参数合同、实际执行和原图保护，不是分割精度或修图自然度的 ground truth。算法性质与真实图片自然度在独立 Rust 测试及 `doc/qa` 图像对账中验证。

`portrait-exif6.jpg` 是原有 EXIF 方向归一化回归输入。
