# image-core 0.2.0

同一套 Rust `apply_graph` 逻辑编译到浏览器 WebAssembly 和原生渲染 CLI，执行 JSON EditGraph。依赖成熟 serde/serde_json、image、wasm-bindgen 与 clap。

当前实现：

- `RgbaImage` 内存模型和输入校验；
- `identity` Operation；
- `color.basic`：曝光、色阶、色温、饱和、纹理等20项；
- `filter.film`：8种风格与确定性颗粒、暗角、辉光、光晕、褪色；
- 禁用 Operation 的 no-op 行为；
- crop、任意 rotate、flip、resize、perspective、网格及推移/收缩/膨胀/双向旋转/重建液化；
- 脸型、五官、身形、衣物的局部参数；
- 皮肤、仿制/修复、传统蒙版填补、独立绘色和擦除；
- 曲线、HSL、胶片模拟和三线性3D LUT；
- 选区、蒙版、冻结保护、透明度和混合模式；
- 参数白名单、类型/范围和输入大小校验。

`ColorAdjustParams` 和 `FilmFilterParams` 使用归一化参数，输入图像保持不可变，所有操作返回新的 `RgbaImage`。胶片颗粒由 seed 驱动的整数哈希生成，因此预览和导出可以复现相同结果。

使用 `make native` 和 `make wasm` 构建。Web 调用 `render_rgba(width,height,pixels,graphJson)` 返回 `{width,height,pixels}`；Go 调用 `image-render --input PATH --output PATH --graph PATH --format png|jpeg --quality 92`。

共享 JSON 协议的滑杆参数均为百分比；早期的 `ColorAdjustParams`/`FilmFilterParams` 归一化 typed API 仅保留兼容，新增宿主应调用 `apply_graph`。

算法、完整参数表、测试与 Demo 限制见 [FUNC-007](../../doc/func/FUNC-007-shared-rust-render-engine.md)。像素处理保持在 Rust/WASM/native worker，不放进 Go API 热路径。
