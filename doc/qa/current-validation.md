# 当前版本验证与已知失败

日期：2026-10-13。本页记录 MLS、分层重塑、Guided Skin 和客户端/服务端 LaMa 的当前版本。旧 phase1-validation 为历史联调证据。**工程检查通过，全部修图功能逐项 8/10 的自然度目标仍未完成。**

## 工程检查

| 检查 | 实际结果与范围 |
|---|---|
| Rust release 全量 | 141 库测试 + 10 public graph 黑盒测试通过；2 ignored 为真实街景手动检查与人脸性能基准，不计通过数 |
| Rust clippy all-targets -D warnings | 通过 |
| native/WASM release | 已构建；传统 graph 使用同一 Rust 核心 |
| Go test/vet | 全部通过；含真实 Rust 混合图、缓存、失败处理、client 节点拒绝与 v3 预处理缓存键 |
| Python LaMa 包装 | 11 项通过；含 float 面积蒙版保留单像素洞、未知 RGB 排除和实际 FastAPI 路由，包装测试不评价模型画质 |
| TypeScript/Vite 生产构建 | 通过 |
| 完整 Playwright | 36/36 通过，独占运行约 2.3 分钟；含所有公开滑杆 native/WASM RGBA 一致、独立本地编辑、模型调度和错误恢复 |
| 本地模型 | MediaPipe、face parser、服务端 TorchScript 与客户端 ONNX 均已安装；权重 SHA 校验及实际推理分别核对 |
| 客户端真实完整 graph | API 请求 0，选区外/alpha 变化 0，40% opacity + 后段与独立 Rust 回放逐字节差异 0，零 opacity 差异 0 |
| 服务端真实完整 graph | 原实测 10.807s/缓存 0.477s；最后 v3 重启实测 11.381s/缓存 0.313s，选区外/保护/alpha 变化 0、RGBA 一致、client 实际返回 422 |

## 真实照片结果

- 人脸：六张重新定位的瘦脸样片，背景、头发、耳朵、饰物变化均 0；同图九参数 ±60 共 18 次回放。真实皮肤轮廓会移动，但向头发外扩受冻结限制、额头加长未完成发际线移动、下巴新边缘暗点和颈部阴影不匹配仍未通过。
- 身体：五图 15 项回放，原已知背景和支持区外变化均 0；主观工程评分 4.10–7.20，全部未达 8。侧身手/发重影、宽衣/包误动、电脑边缘及腿间残影保留。自动 Pose 尚无真实部位像素解析，不能把骨架 polygon 当 partMask。
- 五官/肤色/妆容：三图 × 41 参数共 123 次浏览器批测完成，支持区外与真实场景背景保护检查无失败；皮肤另六图 × 八组参数共 48 次 native 回放，支持区外与 alpha 变化 0。头部轮廓外包含颈部皮肤，不能误作背景；报告保留旧指标并增加 person/skin 语义检查，没有放宽阈值。几何与妆容自然度还需逐项目视，不以零变化或未检出目标计成功。
- 衣物：此前三图 21 案例选区外变化 0；传统细纹弱化有效，大褶皱、衣物结构重建仍未实现，本轮没有新算法实拍验收。
- 去人：两端均真实运行 LaMa，简单街景人物移除比传统 exemplar 干净；拥挤人群轨道错纹、建筑窗格糊块仍失败。建筑是有 clean 参考的合成遮挡实验，不计为真实人物样片。

逐项证据：[人脸](face-contour-visual-review.md)、[身体](body-contour-v2-review.md)、[五官与肤色批测](feature-skin-multisample-review.md)、[皮肤 48 项记录](skin-guided-validation.json)、[皮肤实拍目视](skin-guided-review.md)、[衣物](clothing-multisample-review.md)、[服务端 LaMa](algorithm-audit-lama.md)、[浏览器 LaMa](client-inpaint/browser-client-review.md)。源照片清单在 test/images/manifest.json；全量大 raster graph/output 保存在忽略提交的 test/results/，精简报告与代表图入 Git。

## 性能与完成范围

本机 Intel Mac、32GiB；当前传统核心仍 CPU/WASM + Canvas2D、8-bit RGBA。人脸 768px native CLI 多数约 368–851ms，身体约 640–1188ms；皮肤本组最长边1024px，48项约 229–662ms。均含读图/PNG，不包含重新检测，不能宣称滑杆实时或大图效率达标。

客户端正式隔离服务器使用实际 WASM 四线程：首张完整处理 36.187s，推理 15.155s，复用同候选并修改后段 0.529s。原三图临时服务器只有一线程，首次 50.84–99.43s；不能将这些测量混作相同配置。服务端复杂图 HTTP 22.64–29.15s，重启/JIT/权重加载会增加冷启动。缓存时延不等于新选区推理速度；WebGPU、多浏览器与移动设备未验证。

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
