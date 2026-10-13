# 当前版本验证与已知失败

日期：2026-10-13。本页记录 MLS、分层重塑、Guided Skin 和客户端/服务端 LaMa 的当前版本。旧 phase1-validation 为历史联调证据。**工程检查通过，全部修图功能逐项 8/10 的自然度目标仍未完成。**

## 工程检查

| 检查 | 实际结果与范围 |
|---|---|
| Rust release 全量 | 182 库测试 + 10 public graph 黑盒测试通过；2 ignored 为真实街景手动检查与人脸性能基准，不计通过数 |
| Rust clippy all-targets -D warnings | 通过；含本轮妆容材质回归 |
| native/WASM release | 已构建；传统 graph 使用同一 Rust 核心 |
| Go test/vet | 全部通过；含真实 Rust 混合图、缓存、失败处理、client 节点拒绝与 v3 预处理缓存键 |
| Python LaMa 包装 | 11 项通过；含 float 面积蒙版保留单像素洞、未知 RGB 排除和实际 FastAPI 路由，包装测试不评价模型画质 |
| TypeScript/Vite 生产构建 | 通过 |
| 完整 Playwright | 41 项通过；包含身体 parser/真实部位保护、raster protection、身体缓存和旧帧取消回归。保护断言同时覆盖 raster 与 brush，要求存在非零保护像素且保护像素 RGBA 保持 |
| 本地模型 | MediaPipe、face parser、服务端 TorchScript 与客户端 ONNX 均已安装；权重 SHA 校验及实际推理分别核对 |
| 客户端真实完整 graph | API 请求 0，选区外/alpha 变化 0，40% opacity + 后段与独立 Rust 回放逐字节差异 0，零 opacity 差异 0 |
| 服务端真实完整 graph | 原实测 10.807s/缓存 0.477s；最后 v3 重启实测 11.381s/缓存 0.313s，选区外/保护/alpha 变化 0、RGBA 一致、client 实际返回 422 |

## 真实照片结果

- 人脸：六张重新定位的瘦脸样片，背景、头发、耳朵、饰物变化均 0；同图九参数 ±60 共 18 次回放。v10 统一 physical alpha 参与旧边缘、holes 和 donor 判断，已知 F/B 回归与 72 项保护回放通过。真实皮肤轮廓会移动，但向头发外扩受冻结限制、额头加长未完成发际线移动、耳下 hair/scene 阶梯、强缩短颈部阴影与细边仍未达8；hairline、pure donor、harmonic候选均保留失败并拒绝生产。
- 身体：FASHN 18 类像素解析 + Pose 竞争分区，v14 增加低分辨率 instance 的一像素不确定边恢复，修复真实裤腿边缘被误冻成黑点，同时保留两像素间隙排除和椅背保护；五图15项、`-60/-30/+30/+60` 共60次真实回放及 3 项浏览器测试通过。腰、上臂、腿的边界归属有所改善，但宽衣/外套同类归属、腿间窄带与细色边仍保留，未把局部修复外推为 8 分。
- 五官/肤色/妆容：三图 × 41 参数共 123 次浏览器批测完成，支持区外与真实场景背景保护检查无失败；皮肤另六图 × 八组参数共 48 次 native 回放。新增自适应黑眼圈候选用眼下 polygon 和周围皮肤估计局部光照，六图 × 4 强度共24项，零值/选区外/alpha 均保持，热运行中位数约218ms（683×1024），真实图自然度暂评约7.4；它不能替代眼袋/泪沟结构重建。脸部 demix6 只减弱下巴缩短旧颈部混色边，六图 chin/jaw 正负极值48项保护通过，但该子项约6.5，仍未达8。
- 五官/肤色/妆容：三图 × 41 参数共 123 次浏览器批测完成，支持区外与真实场景背景保护检查无失败；皮肤另六图 × 八组参数共 48 次 native 回放。新增自适应黑眼圈候选用眼下 polygon 和周围皮肤估计局部光照，六图 × 4 强度共24项，零值/选区外/alpha 均保持，热运行中位数约218ms（683×1024），真实图自然度暂评约7.4。妆容材质 v1 新增三图 × 2 类 × 5 强度共30项，牙齿 polygon、选区外和 alpha 保护通过，唇色/眉色暂评7.2；它仍不能替代眼袋/泪沟、口红材质或毛发级眉妆重建。
- 衣物：此前三图 21 案例选区外变化 0；传统细纹弱化有效，大褶皱、衣物结构重建仍未实现，本轮没有新算法实拍验收。
- 手工修复：新的 `repair.heal` 使用有限环形 patch 匹配和谐波光照补偿；三张痤疮照片的明确小范围白头/小红痘圈选结果约7.8，深层囊肿约6.8。圈外、alpha 与透明 RGB 保持不变；不外推为自动祛痘或全脸瑕疵分类已完成。
- 去人：两端均真实运行 LaMa，简单街景人物移除比传统 exemplar 干净；拥挤人群轨道错纹、建筑窗格糊块仍失败。MAT/MI-GAN/PatchMatch候选均实跑且未通过，保持生产LaMa。建筑是有 clean 参考的合成遮挡实验，不计为真实人物样片。

逐项证据：[人脸](face-contour-visual-review.md)、[身体V4](body-parts-v4-review.md)、[身体V3](body-parts-v3-review.md)、[五官与肤色批测](feature-skin-multisample-review.md)、[皮肤48项](skin-guided-review.md)、[真实痘印挑战](skin-acne-v3.json)、[蒙版效率](mask-efficiency-review.md)、[衣物](clothing-multisample-review.md)、[服务端LaMa](algorithm-audit-lama.md)、[浏览器LaMa](client-inpaint/browser-client-review.md)。源照片清单在 test/images/manifest.json；全量大 raster graph/output 保存在忽略提交的 test/results/，精简报告与代表图入 Git。

## 性能与完成范围

本机 Intel Mac、32GiB；当前传统核心仍 CPU/WASM + Canvas2D、8-bit RGBA。人脸 768px native CLI 多数约 368–851ms，身体约 640–1188ms；皮肤本组最长边1024px，48项约 229–662ms；自适应黑眼圈专项全图 native 总进程约177–864ms，graph 热运行中位数约218ms。均含读图/PNG的项目会单独标明范围，不包含重新检测，不能宣称所有滑杆实时或大图效率达标。

客户端人体解析实际WASM四线程冷启动约26.02s（load7.10s、inference18.25s），滑杆不重新解析；去人客户端首张完整处理36.187s、候选复用0.529s；服务端复杂图HTTP22.64–29.15s。mask.rs等价优化在4组1024px隔离回放中热中值下降23%–64%、RGBA逐字节相同；不等于浏览器全链路效率验收。WebGPU、多浏览器与移动设备未验证。

失败与不可用功能完整保留。未实现结构项继续明确拒绝，传统手工工具可独立使用；普通导入不上传，显式连接 AI 或选择 server 才执行相应上传。当前画质不足的根因、评分规则与后续目标见 [逐功能目标](quality-targets.md)。

## 重现

```sh
make setup
make client-models       # 可选浏览器去人权重
make background-models   # 可选服务端环境与权重
make background         # 另一个终端保持运行
make check
make background-test
make dev
make e2e
node scripts/check_client_graph.mjs
node scripts/check_client_inpaint.mjs
make quality-face
make quality-body
make quality-skin
```

真实图 quality 脚本保留失败与未验证；成功执行或导出数量不等于自然度通过数量。make quality 的历史批测与本页当前局部专项报告应按版本区分。
