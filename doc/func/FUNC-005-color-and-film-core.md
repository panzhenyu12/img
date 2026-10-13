# FUNC-005 调色与胶片模拟

状态：已验证（Web WASM / Go native，2026-10-12）

关联需求：COLOR-001..014（当前8-bit调色子集）；FILM-001..009、FILM-011..013、FILM-016、FILM-017、FILM-019。颗粒当前只有强度与确定性seed，不将 FILM-010 的大小粗细或 FILM-014 的噪声分离列为完成。

## 用户行为与实现

打开光影/曲线/HSL/胶片工具，拉滑杆、添加曲线点、选择颜色通道或胶片风格。每个效果可开关、排序、重置、调透明度与蒙版。胶片有当前照片的真实缩略图、分类搜索，自定义命名/复制/保存；支持导入 `.cube` LUT。

前端：registry、ToolPanel、lut.ts、Worker；后端：Rust color.rs、mask.rs，Go调native。禁止用 JS 和 Go 重新实现颜色算法。

## Operation 与参数

- color.basic：20项基础参数；百分数，blackPoint/whitePoint为0..100、gamma0.1..5，blackPoint必须小于whitePoint。
- color.curves：`{channel:'rgb'|'red'|'green'|'blue',points:[{x:0,y:0},{x:1,y:1}]}`。
- color.hsl：hsl/channels按色相中心、范围、shift/saturation/lightness；UI提供8色分区。
- filter.film：`{preset:'classic',intensity:70,grain:20,vignette:10,glow:0,halation:0,fade:0,seed:42}`。
- filter.lut：`{size,data,domainMin,domainMax,amount}`，title仅用于UI标签；2..65立方，红轴最快，三线性采样。

## 完成标准与测试

Rust验证曲线/LUT identity、HSL和胶片alpha/确定性；Playwright真实执行所有注册滑杆并断言像素变化，测试曲线、HSL、胶片缩略图/预设、LUT强度、排序与PNG/JPEG导出。seed保证同工程重算不随机跳变。

## 当前限制与变更

自定义风格不是富士官方算法。当前8-bit sRGB处理无ICC/16-bit/HDR；预设保存浏览器localStorage，不云同步。2026-10-12 用shared Rust engine取代早期JS/Go颜色路径，统一Web参数百分数；旧Rust `-1..1` API不作为共享协议。
