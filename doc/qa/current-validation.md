# 当前版本验证与已知失败

2026-10-15 后续增量：唇明亮 V2 已更新本地 native/WASM，Rust完整回归209+10+5通过、3忽略；浏览器/API内含0、±30、±100的1项回归通过，Web构建及实际HTTP产物哈希通过。五原照55组与等亮度对照支持减少漂白，但强负向唇缘仍失败，不恢复整体评分。最新颜色源码/产物见[唇亮度接入记录](lip-lighting-v2-integration-20261015/review.md)；下方v19工程表是历史快照，不能当作本次重复运行。

2026-10-15 最新增量：物理边缘残点修复已部署本地共享核心，209单测+10契约、Web构建、9档原图网页/Go/native/WASM RGBA对账通过；总体自然度仍未验收。见[本轮接入记录](physical-fringe-v2-integration-20261015/review.md)。

日期：2026-10-14。本页记录现行v19脸型/V4b五官共享核心与历史研发快照。旧phase1-validation与V4b checkpoint保留为历史证据。**工程检查和像素权限通过不代表自然度通过；全部修图功能逐项8/10仍未完成。** 当前源码/产物hash和检查日志见[v19生产集成冻结](face-joint-v19-integration/validation.json)，此前V4b记录见[旧checkpoint](algorithm-checkpoint-20261014/validation.json)。

2026-10-14用户原图最新反馈：瘦脸正负方向出现轮廓裂缝和连接断开，五官/妆容被报告不生效。相关现行评分已撤销，按不可用/未通过处理；以下历史工程和样片证据不能覆盖这个失败。见 [用户案例](user-face-contour-20261014-review.md)。

最新集成状态：v19联合脸颈已按用户授权接正式，固定QP/竖向跨零连续及混合保护完成；新默认四档UI/Go回归和原照四档UI/API/native/WASM/旧candidate全RGBA一致，受影响10项E2E通过。**用户随后明确拒绝瘦脸审美：不自然、曲线不好看，脸/五官/身体不能统一曲线。** 接入验证通过不等于质量验收，耳侧12px原skin岛/硬边、单原片和多图自然度缺口仍保留，不升8分，见[生产集成对账](face-joint-v19-integration/review.md)。

## 工程检查

| 检查 | 实际结果与范围 |
|---|---|
| Rust release 回归 | 最终neck gate源码face范围42 passed、0 failed、1 ignored；稍早全仓206库+10 public graph通过属于gate微调前集成快照，不宣称最终source全量重跑。历史V4b195+10及86场景/9跨零单独保留，不累加内部场景成测试数 |
| Rust clippy all-targets/all-features -D warnings | 当前源码通过 |
| native/WASM release | 当前v19已构建并冻结，Web/Go共用Rust；native f659ae4c…，WASM 96a890d8…，完整hash见当前validation。隔离v14–v18/颈层重建仍拒绝，旧V4b worker保留历史 |
| Go test/vet | 当前源码全部通过；额外执行 go test -count=1 ./... 避免外部 native 更新被 Go 测试缓存掩盖，含真实 Rust 混合图、缓存与失败处理 |
| Python LaMa 包装 | 11 项通过；含 float 面积蒙版保留单像素洞、未知 RGB 排除和实际 FastAPI 路由，包装测试不评价模型画质 |
| TypeScript/Vite 生产构建 | 通过 |
| Playwright | 当前v19受影响10项全部通过，2.8分钟；含新joint（1项内四档UI/Go）、真实MediaPipe检测、4项context、raster保护/公开控制parity。未重跑完整49项。此前48项4.6分钟和V4b8项22.7秒属于各自历史快照 |
| v19原照UI/API parity与保护 | 一张1024² canonical原照±30/±100：UI/WASM、GoAPI、immutable native与冻结candidate全RGBA差异0，page errors0、alpha0变化；实际浏览器耳/发/配饰coverage≥.5、原ROI外、y≥700低颈变化0。四数值不作四独立原照 |
| V4b照片/alpha历史专项 | 原eyeTilt+100 UI逐byte一致、5项raw非均匀alpha一致、48次未受影响输出hash不变保留历史报告；geometry/color源码hash保持冻结，raw核心合同不等于Canvas导入hidden RGB无损 |
| 本地模型 | MediaPipe、face parser、服务端 TorchScript 与客户端 ONNX 均已安装；权重 SHA 校验及实际推理分别核对 |
| 客户端真实完整 graph | API 请求 0，选区外/alpha 变化 0，40% opacity + 后段与独立 Rust 回放逐字节差异 0，零 opacity 差异 0 |
| 服务端真实完整 graph | 原实测 10.807s/缓存 0.477s；最后 v3 重启实测 11.381s/缓存 0.313s，选区外/保护/alpha 变化 0、RGBA 一致、client 实际返回 422 |

## 真实照片结果

- 人脸历史：v11只修局部下颌采样，六图/九参数±60/72保护回放保留原范围。v14–v18仍隔离拒绝：v17的39通过不改变六档失败输出，v18的37通过/3失败把+100内收约20px削弱至8px；独立颈层延拓/LaMa仍材质硬接。见[实验对账](../func/FUNC-011-face-rbf-neck-experiments.md)与[材质负例](face-amodal-neck-20261014/review.md)，不因v19接入追认通过。
- 当前联合脸颈v19：已正式接入共享native/WASM，真实Web/UI/API四档保持旧candidate逐字节效果，大黑楔与斜接减少，+100旧源face polygon约20.075px测量保留在隔离报告，不当作本轮新量化。旧40 face测试是隔离快照；最终gate face42通过、新默认joint与受影响10E2E通过。四档真实浏览器冻结/ROI外/低颈/alpha变化0，仍有耳侧12px原skin岛、细点、硬边和单原照局限，且用户明确拒绝当前不自然/曲线不好看的审美。工程接入不等于质量验收，不恢复8分；当前证据见[生产集成](face-joint-v19-integration/review.md)、[实际Web对照](face-joint-v19-integration/actual-web-comparison.png)，原实验见[旧v19](face-joint-v19/review.md)。
- 身体：FASHN 18 类像素解析 + Pose 竞争分区，v14 增加低分辨率 instance 的一像素不确定边恢复，修复真实裤腿边缘被误冻成黑点，同时保留两像素间隙排除和椅背保护；五图15项、`-60/-30/+30/+60` 共60次真实回放及 3 项浏览器测试通过。腰、上臂、腿的边界归属有所改善，但宽衣/外套同类归属、腿间窄带与细色边仍保留，未把局部修复外推为 8 分。
- 五官/肤色/妆容：已修复工具切换/工程恢复丢检测上下文；统一语义五官连续流、眼睑包络与连续冻结域修正眉峰/眉尾拉弯眼睑、硬边界折叠和跨零算法跳变。V4b另修旋转误用有限位移造成的眼中心约9.34%额外放大、identity丢透明RGB；实际眼缘polygon面积仍有约0.18%–1.93%减小，不宣称全眼严格刚体。六张独立原照眉毛对照已查看，强眉峰夸张、眉尾作用受限与邻近纹理拉伸仍保留；这些修复不等于全部五官/妆容达8。见[FUNC-020](../func/FUNC-020-feature-svf-quality.md)与[V4b复核](brow-flow-v4-20261014/hotfix-v4b/review.md)。历史123次浏览器/48次皮肤/30项妆容记录仅为相应快照；唇色材质、鼻梁响应弱、眼袋/泪沟和毛发级眉妆仍未解决，见[原始失败](user-feature-audit-20261014/review.md)。
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
