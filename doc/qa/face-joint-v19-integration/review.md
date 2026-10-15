# v19 联合脸颈生产集成对账

状态：v19已按用户授权接入正式Rust共享核心，native/WASM构建、四档原照真实UI/API回放与受影响10项E2E均通过。此前`face-joint-v19`的40个face测试与四档原照是隔离快照，不能当作本页新生产版本已通过的记录。质量仍未达逐功能8分。

**用户随后明确拒绝当前审美效果：瘦脸不自然、曲线不好看，脸/五官/身体曲线需要各自处理。接入验证通过不等于质量验收。** 当前部署/冻结证据保留，后续候选须另存，不覆盖本轮结果；本页不恢复数字评分。

## 已确定的实现范围

可信脸和相邻可见上颈作为同一物理前景、同一速度流传输，低颈回落到静止；需要face raster、faceOutline、faceContour及对应faceContourMask、skin/neck像素连通。缺项、零neck或断开的neck保留原有效face-only/手工路径。耳/发/配饰、明确保护及不相关人物保持独立阻断。

neck gate要求属于owned连通域、`faceContourMask≤.5`、`skin>.5`且`neck>.8`的真实外脸颈像素；空skin或仅在脸内误标的neck不能激活联合域。默认测试分别生成与faceOutline/faceContour匹配的head/face raster，皮肤与neck在脸缘外保持真实像素连通。

QP目标拟合使用固定路由，chin/forehead的微小数值不再通过旧零点条件开关整个QP方法；这些参数的竖向位移连续保留。混合模式、透明度和软选区合成也需保留semantic冻结/用户保护，而不只在normal输出中成立。没有新增公开experiment参数、环境开关、前端控件或Go像素算法；手工与AI/MCP仍共用`face.adjust`。

QP的侧向拟合在source-v坐标排列，保留source-v再叠加解析dv插值以使零点连续。因此其侧曲线单调/曲率约束不能证明混合chin/forehead后的最终物理外曲线绝对没有新凹凸或fold。normal保留候选的一次几何衰减；non-normal才追加最终softselection/protect权重并恢复identity硬冻结像素，不把两种合成当作同一个严格几何map。

## 本轮结果与范围

- 默认`face-joint-parity.spec.ts`：复用跟踪的真实检测polygon坐标，生成faceContour/.5边界对应的raster、连续skin/neck及纹理。真实UI导入工程、四档脸宽控件、保存工程和PNG导出，与Go native API逐RGBA字节核对；检查neck实际变化、alpha及完整保护，不需要本机忽略照片/模型。
- 可选原照专项：同一1024 canonical PNG和原graph、±30/±100，真实UI→PNG导出、Go API、immutable生产CLI/WASM、旧冻结v19candidate全RGBA对账。记录source、请求/保存graph、源码、Cargo文件、binary、WASM和各PNG的哈希，起止保持不可变。
- Root统一运行当前Rust、clippy、Go、Web及受影响E2E；日志与`validation.json`和本页报告均区分历史与当前版本。

| 检查 | 本轮实际结果 |
|---|---|
| 最终neck gate源码face测试 | 42 passed、0 failed、1 ignored；不是全仓新206项重跑 |
| gate最后微调之前全Rust | 206库测试+10 public graph测试通过；这是稍早的集成快照，不能标成最终source全量重新验证 |
| 最终clippy、native/WASM | clippy通过，`make native wasm`通过并冻结 |
| Go与Web | fresh `go test -count=1 ./...`、vet及新WASM后Web生产构建通过 |
| 新默认joint E2E | 1测试内±30/±100实际UI/Go四档通过；neck实际像素变化、hard保护、alpha与全RGBA parity均通过 |
| 原照四档 | UI/WASM↔immutable native、GoAPI↔native、旧v19candidate↔native：每项全1024²RGBA差异0、alpha变化0、page errors0 |
| 源码/输入/worker不可变 | 起止核对7个Rust/Cargo文件、原graph、canonical PNG、binary、WASM和4 candidate输出hash不变 |
| 受影响E2E | 10 passed、0 failed，2.8分钟；包含新joint、raster保护/全部公开控制parity、真实MediaPipe检测与4项人像context，不声称重跑完整49项 |
| 原照实际浏览器保护 | 四档导出对耳/发/配饰coverage≥.5、原ROI外、y≥700低颈及alpha的变化均0 |

原照结果见[完整UI/API/CLI/WASM与hash报告](ui-parity.json)和[本次运行日志](canonical-ui.log)。PNG压缩编码允许不同；比较的是解码后的全部RGBA。四个native/API/candidate PNG各自连文件SHA也一致，正式集成保持冻结候选的四档效果。

实际浏览器导出保护检查见[actual-ui-contracts.json](actual-ui-contracts.json)，统一构建/测试范围与全部当前哈希见[validation.json](validation.json)。

| 脸宽数值 | 相对原图改动像素 | UI导出墙钟ms |
|---:|---:|---:|
| -100 | 25668 | 2566 |
| -30 | 18062 | 3457 |
| +30 | 18166 | 5561 |
| +100 | 26030 | 2562 |

这些时间包含UI/编码与并发负载，不是热帧或实时benchmark。

## 已知限制与评分

冻结实验的同原照四档大黑楔和脸颈斜接明显改善，强档源face polygon约20px运动，但仍有耳侧12px遗漏原skin岛、耳根细点和硬边。只有一张独立原照；四个数值不增加原照数。后续双端字节一致只能证明路由/执行正确，不能替代多原片、供体权限和自然度验收，不恢复8分或旧数字评分。原照与旧candidate保持不改，原尺寸效果见[独立实验复核](../face-joint-v19-root-review/review.md)。

实际Web四档导出已由主任务目视复核，[Original / -100 / -30 / +30 / +100对照](actual-web-comparison.png)显示脸颈联动与已接受候选一致；+100耳侧细点、原皮肤岛和硬边仍可见。本专项只覆盖faceWidth四档，不能将其外推到全部脸型/下巴/额头参数的视觉验收。

## 当前冻结

| 产物 | SHA256 |
|---|---|
| face.rs | `5e4ccd45be6d1273e8de96f7e061ec5319d3d163fe716ee49ff29c5a4b8c6650` |
| face_flow.rs | `410d4b531945476c31fba33633723f22d428bc10492c24dc0b67687656be5696` |
| native | `f659ae4c047757b940a5acc103c6f8874bffb790f19cd7e515bbd27e1147b9b5` |
| public WASM | `96a890d856efbc4e54bed8d7b0ed880539ad93a5bffa064e9c9527b612ad50cf` |

geometry V4b和skin warmth颜色源码hash保持既有冻结值。所有源graph/源PNG/请求及保存graph/导出PNG、Cargo manifest/lock hash保留在UI报告，不把同一原照多种强度当作新独立样本。

## 复现

```sh
cd apps/web
npx playwright test tests/face-joint-parity.spec.ts
```

可选原照先由主任务构建并冻结当前worker，再在仓库根目录运行：

```sh
FACE_JOINT_QA_BINARY=/absolute/path/to/immutable/image-render node scripts/audit_face_joint_production_ui.mjs
```

默认测试的合成纹理和真实检测坐标不作为审美golden；专项不能脱离本机原照缓存运行，不把它当成默认CI必需品。
