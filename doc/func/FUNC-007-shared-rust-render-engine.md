# FUNC-007 共享 Rust 图像渲染引擎

状态：已验证（一期 Demo，CPU/WASM 与 native 共用算法）

关联需求：`CORE-007` 原图不可变、`CORE-011`/`CORE-012` 效果开关；`FACE-001` 至 `FACE-019` 手工脸型；`SKIN-001` 至 `SKIN-004`、`SKIN-014` 至 `SKIN-016`；`BODY-001` 至 `BODY-012`；`CLOTH-001` 至 `CLOTH-012`；`WARP-001` 至 `WARP-005`、`WARP-008` 至 `WARP-010`；`REPAIR-001` 至 `REPAIR-007`；`COLOR-001` 至 `COLOR-014`；`FILM-004` 至 `FILM-013`；`MAKEUP-001` 至 `MAKEUP-005`、`MAKEUP-011`。

关联需求描述的是完整产品目标。本页的完成标准和限制记录一期实际算法范围，不能据此推断所有完整产品需求已经达到专业修图质量。

## 用户行为

用户导入照片后，拖动参数、移动控制点、使用画笔或提交 AI 编辑提案。操作记录在统一 EditGraph 中；浏览器从源 RGBA 像素重新执行图，后端从原文件执行相同图。关闭 Operation、撤销或清空操作后，原始照片始终可恢复。

## 前端实现

WASM 通过 `wasm-bindgen` 导出：

```ts
render_rgba(width: number, height: number, pixels: Uint8Array, graphJson: string)
  // returns { width: number, height: number, pixels: Uint8Array }
engine_version() // image-core-0.2.0
```

前端在 Web Worker 中调用，操作参数直接沿用 JSON 协议。算法不依赖网络、AI 模型或 Go API。错误以 JS 异常返回，宿主保留上一帧预览和原图。

产物由以下命令生成到 `apps/web/public/wasm/`：

```bash
make wasm
```

## 后端实现

Go 使用 Gin/Viper 接收请求和管理临时文件，通过原生 CLI 调用 Rust：

```bash
core/image-core/target/release/image-render \
  --input /tmp/source-image \
  --output /tmp/output.png \
  --graph /tmp/edit-graph.json \
  --format png \
  --quality 92
```

成功时 stdout 输出一行 JSON：

```json
{"width":820,"height":1024,"format":"png","output":"/tmp/output.png"}
```

失败时 stderr 输出原因并以非零状态退出。CLI 根据图片 magic bytes 解码，可使用无扩展名输入文件；通过成熟 `image` decoder 读取并应用 EXIF 方向，与浏览器的视觉方向对齐。解码器最大分配128MB；编码成功且输出流 flush 后才输出成功元数据。PNG 保留 alpha；JPEG 将透明区域合成到白色背景，quality 为 `1..100`。重新编码保证渲染像素一致，不保证与原文件二进制完全相同。

Rust 库统一入口：

```rust
pub fn apply_graph(source: &RgbaImage, graph: &EditGraph) -> Result<RgbaImage, ImageError>
pub fn apply_graph_json(source: &RgbaImage, graph_json: &str) -> Result<RgbaImage, ImageError>
```

## Operation 类型与参数

共享 Operation 使用 `type`、`params`、`enabled`、`controlPoints`，附加的来源、模型、说明等 envelope 元数据可保留在宿主协议中。所有滑杆采用 `-100..100` 或 `0..100`；几何坐标、半径和区域采用归一化坐标，裁剪和图像尺寸采用像素。

| 类型 | 已执行参数 |
|---|---|
| `identity` | 无；保持源像素 |
| `crop` | `x,y,width,height`，像素；裁剪范围限定在源画布 |
| `rotate` | `degrees`；`expand` 默认 true；支持任意角度和透明扩边 |
| `flip` | `axis: horizontal/vertical` |
| `resize` | `width,height` 像素；成熟 `image` 库的 Lanczos3 重采样 |
| `perspective` | `corners` 或 `points` 四个 `{x,y}`，顺序左上、右上、右下、左下；真实单应变换 |
| `warp.grid` | Operation `controlPoints` 或 params `points/controlPoints`；`inX,inY,outX,outY`；逆距离加权逆映射 |
| `warp.liquify` | `strokes:[{x,y,toX,toY,radius,strength,mode,direction?}]`；mode 为 push/pinch/bloat/twirl/reconstruct；twirl direction 为 `-1/1` |
| `face.adjust` | `region` + `adjustments`；可选 `landmarks:[{x,y,toX,toY,weight}]` 点对，见下表 |
| `feature.adjust` | `region` + `adjustments`；可选 `landmarks` 点对 |
| `body.adjust` | `region` + `adjustments`；可选 `bodyPart`、`landmarks` 与 Pose polygon mask；按腰、胯、四肢等语义区域回放 |
| `clothing.adjust` | `region` + `adjustments`；可选 `landmarks` 点对；`smooth` 使用低频/高频分离降低褶皱，`texture` 控制织物细节保留 |
| `skin.retouch` | `smooth,texture,whiten,warmth,redness`；边缘感知柔化、纹理保留、提亮、色温和去红 |
| `repair.clone` | `strokes:[{x,y,sourceX,sourceY,radius,strength}]`；源纹理仿制、柔和边缘 |
| `repair.heal` | 同上；source 坐标可省略，自动选邻近低差异样本；边界颜色校正融合 |
| `repair.inpaint` | `region/mask` 选区；`radius` 默认 .05；`iterations:1..64` 默认16；传统近邻传播与边界松弛填补 |
| `paint.brush` | `color:[r,g,b]` 各 `0..255`；`strokes:[{x,y,radius,strength,erase?}]`；独立透明绘色层 |
| `color.basic` | `exposure,brightness,contrast,highlights,shadows,whites,blacks,temperature,tint,saturation,vibrance,clarity,texture,dehaze,sharpness,noise,blackWhite,blackPoint,whitePoint,gamma` |
| `color.curves` | `points:[{x,y}]` 归一化；`channel:rgb/red/green/blue`；分段线性颜色查表 |
| `color.hsl` | `hsl:[{hue,range,shift,saturation,lightness}]`；hue `0..360`、range `1..180` 度，其余百分比 |
| `filter.film` | `preset,intensity,grain,vignette,glow,halation,fade,seed`；自定义风格 warm/cool/classic/fade/vivid/mono/cinema/neutral，并兼容 pastel/night |
| `filter.lut` | `size:2..65,data:RGB[size³],domainMin,domainMax,amount`；RGB 红通道最快的三线性插值 |

`color.basic` 曝光 `-100..100` 映射到 `±2 EV`；色阶 `blackPoint/whitePoint` 为 `0..100`，默认 `0/100`，blackPoint 必须小于 whitePoint；gamma 为 `.1..5`，默认1。`blackWhite` 可用 boolean 或 `0..100`。透明度不随普通调色、磨皮、修复和 LUT 改变。

局部参数白名单与 UI registry 一致，未知 key 会明确报错：

| 分类 | adjustments keys |
|---|---|
| 脸型 | forehead, foreheadWidth, temples, cheekbones, jaw, chin, chinWidth, chinPoint, faceWidth, faceHeight, midface, doubleChin, symmetry |
| 眼睛 | eyeSize, eyeWidth, eyeHeight, eyeSpacing, eyeTilt, eyeCorner, eyeInnerCorner, eyeOuterCorner, eyelid, eyeBag, darkCircle, tearTrough |
| 眉毛 | browLength, browWidth, browThickness, browPeak, browTail, browColor, browSpacing |
| 鼻子 | noseBridge, noseWings, noseLength, noseTip, noseHeight, noseStraight, nostril, noseRoot, noseBase, noseLipAngle |
| 嘴唇 | mouthWidth, mouthHeight, upperLip, lowerLip, lipColor, philtrum, teethWhiten, lipPeak, lipCorner, lipBrightness, lipSaturation, symmetry |
| 身形 | neck, neckLength, shoulder, shoulderLevel, trapezius, collarbone, waist, belly, arms, forearms, legs, thighs, calves, ankles, hips, height, legLength, posture, headRatio |
| 衣物 | smooth, texture, hue, saturation, brightness, fit, edges, sleeves, skirt, collar, cuffs, sleeveLength |

`texture` 用来调节 smooth 的纹理保留程度，单独改变 texture 不会凭空重建纹理。正值 `faceWidth` 表示瘦脸，命名为 width/height 的其它局部参数正值表示扩大相应特征；局部身体纤细参数正值表示收窄。对称功能在选择区域内融合左右镜像，100% 时区域两侧共享平均颜色。

液化 `reconstruct` 笔刷按 strength 将笔刷区域混合回当前液化层的输入图，因此保留更早节点的调色和修图结果；不会直接覆盖原图。对称液化由 Web/AI 提交显式镜像 strokes，`x` 和 `toX` 沿中线镜像；镜像 twirl 将 direction 反号。重建、正反旋转和镜像笔画都经过同一保护蒙版合成。

## 通用蒙版与合成

```json
{
  "region":{"x":0.1,"y":0.1,"width":0.8,"height":0.8},
  "mask":{
    "shape":"ellipse",
    "region":{"x":0.2,"y":0.2,"width":0.5,"height":0.6},
    "invert":false,
    "feather":15,
    "strokes":[{"x":0.5,"y":0.5,"radius":0.05,"strength":100,"erase":true}]
  },
  "protectMask":{"shape":"brush","strokes":[{"x":0.8,"y":0.5,"radius":0.1,"strength":100}]},
  "opacity":70,
  "blendMode":"normal"
}
```

shape 支持 rect、ellipse、polygon、brush、linear、radial。Polygon 使用 `points`；linear 使用 `from/to` 两个归一化点；radial 使用 region 的中心到外缘渐变。所有 shape 都支持追加/擦除 stamps；`invert` 反选；feather 为 `0..100`，映射到短边最大20%的过渡宽度。

支持 normal、multiply、screen、overlay、soft-light、darken、lighten 混合。采样和透明合成使用预乘 alpha 中间计算，避免透明边缘黑边。全保护区域逐像素保持原输入，包括 alpha。

改变画布尺寸的 crop、resize、expand rotate 无法与原尺寸选区合成；若携带局部 mask、protectMask、非100 opacity 或非 normal 混合，会明确拒绝，UI 应隐藏相应设置。同尺寸操作可使用通用合成。

## 浏览器渲染调度与取消

`apps/web/src/core/engine.ts` 在同一时刻只给 Worker 发送一个任务：

```ts
renderImage(source, graph, {
  signal: abortController.signal,
  kind: 'preview' // preview | export | background
})
```

预览队列只保留最新参数；新的 preview 或 AbortSignal 会终止正在执行旧 preview 的 Worker，从而真正停止同步 WASM CPU 像素循环。队列中的 export 按顺序优先执行，持有请求时的操作 JSON 快照，不会被滑杆预览替代。Background 用于滤镜缩略图等任务；排队的取消任务会安全移除。

显式取消 active export/background 会将其 Promise 以 AbortError 拒绝，底层任务运行结束后再继续队列；预览取消会直接替换 Worker。Worker 顶层错误和消息传输错误安全拒绝全部挂起任务，下一次调用重新建立 Worker。WASM/模块网络加载错误返回 `resetWorker:true`，只拒绝当前任务并替换 Worker，防止失败的 ES module import 缓存阻塞后续重试；普通 Operation 参数错误不替换 Worker。

只有真正发送的任务复制并 transfer 源像素，避免快速滑杆每次排队都复制大图。`getRenderSchedulerState()` 暴露当前 active kind、最新 preview、排队 export/background 数量以及 Worker generation/cancelledPreview 计数，便于并发验收。

验收场景：长时间 preview 后提交新 preview，旧 Promise 应以 AbortError 拒绝、Worker generation 增加；export 期间快速提交多个 preview，export 必须成功且只有最新 preview 生效；模块请求临时失败后恢复请求，后续任务应在新 Worker 成功。对应 Playwright 由 Web 集成验收维护。

## 当前限制

- 局部五官和身形基于手工 region 内的二维预设位置，适合正面或近正面的 Demo；不等同于三维面部/人体重建，侧脸需要手工重定位或控制点调整。
- 对称采用左右镜像融合，会混合原来的纹理，尚非基于关键点的几何对齐。
- 祛皱、鼻高、肩平、锁骨等为局部二维几何或颜色近似，无法恢复真实立体材质。
- 修复 heal 是传统颜色校正采样；inpaint 是近邻传播/边界松弛，无法生成被遮挡的人物或复杂场景。AI 生成式修复应由独立 AI 服务产出图层或源素材。
- CPU/WASM 是可运行基线，尚无 WebGPU/wgpu shader 执行器；大图大量笔画的实时性能仍需优化。
- 目前使用 8-bit RGBA；暂无 RAW、完整16-bit、ICC管理或HDR链路。sRGB显示空间下使用统一像素输入，PNG/JPEG导出。

## 完成标准与测试证据

- [x] 原生 CLI 和浏览器 WASM 调用同一个 `apply_graph`。
- [x] 输入图像不可变；Operation 顺序、enabled 和参数校验一致。
- [x] 未知 type/key、数字类型错误、退化透视、坏 LUT 数据明确报错，disabled 操作也先校验。
- [x] 所有局部 registry key 具有真实像素效果；texture 与 smooth 一同测试。
- [x] 裁剪、翻转和90度旋转精确像素测试；任意旋转扩边和透视测试。
- [x] Alpha 保持、保护区域精确保留、蒙版反选和 brush 擦除测试。
- [x] 无 source 修复去均匀背景瑕疵；inpaint 改变目标区域并保留保护区域。
- [x] 绘色橡皮恢复底图，透明底图的半透明绘色保持颜色。
- [x] 对称操作将左右镜像误差从非零降低到零。
- [x] 重建恢复当前液化层输入并保留更早调色；twirl 正反方向、冻结保护逐像素测试。
- [x] `cargo test` 31 个测试通过；`cargo clippy --all-targets -- -D warnings` 通过。
- [x] Native release 与 `wasm32-unknown-unknown` release 编译通过，生成 Web wasm-bindgen 绑定。

输入限制：最大 32M 像素、256个 Operation、1024个控制点/曲线点、每操作4096个 strokes、LUT最大65³。尺寸乘法使用 checked arithmetic，避免 buffer 长度溢出。

## 实现位置

- [engine.rs](/Users/modao/work/img/core/image-core/src/engine.rs)：EditGraph、Operation 顺序及统一合成。
- [validation.rs](/Users/modao/work/img/core/image-core/src/validation.rs)：类型、参数 key、范围和大小边界。
- [geometry.rs](/Users/modao/work/img/core/image-core/src/geometry.rs)：裁剪、采样、几何变形、局部几何与修复。
- [color.rs](/Users/modao/work/img/core/image-core/src/color.rs)：调色、皮肤、局部补色、胶片、LUT。
- [mask.rs](/Users/modao/work/img/core/image-core/src/mask.rs)：选区、蒙版、冻结保护和混合模式。
- [paint.rs](/Users/modao/work/img/core/image-core/src/paint.rs)：独立绘色层与橡皮。
- [wasm.rs](/Users/modao/work/img/core/image-core/src/wasm.rs)：浏览器入口。
- [image-render.rs](/Users/modao/work/img/core/image-core/src/bin/image-render.rs)：原生渲染 CLI。

## 当前进度与变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-12 | 从 identity 骨架扩展到共享 JSON EditGraph 渲染引擎，引入 serde/serde_json、image、wasm-bindgen、clap；完成 native/WASM、真实像素操作、参数校验、29个测试和编译产物。 |
| 2026-10-12 | 增加浏览器单任务调度、最新预览合并、AbortSignal CPU取消、export/background独立队列、模块加载失败 Worker重建；Web生产构建通过。 |
| 2026-10-12 | 液化新增 reconstruct 与 twirl direction ±1；增补恢复/早期节点保留/相反旋转/冻结保护测试，31个Rust测试通过，再次生成native/WASM产物。 |
| 2026-10-12 | 浏览器 decodeImage 收束 Canvas 创建/读取异常，并在成功或失败时释放 blob URL；三种路径 smoke 检查与 Web 构建通过。 |
