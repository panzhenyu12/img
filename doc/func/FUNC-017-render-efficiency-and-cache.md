# FUNC-017 蒙版效率与身体滑杆缓存

日期：2026-10-13。该项提升交互效率，像素质量评分仍由各功能自己的实拍QA判断。

状态：已实现，生产跨端回归通过。关联需求：CORE-007原图不可变、CORE-011/012效果开关，BODY系列部位编辑；复用FUNC-007/008的Operation与参数，没有新增协议字段。

用户行为：连续拖动同一身形滑杆时只显示最新值；切换原图或目标部位重新计算。前端实现位于engine.ts/render.worker.ts与Rust WASM；Go调用同一native核心，无专用缓存协议。完成标准为输出等价、旧帧不显示、不同输入不能误命中缓存、原图和导出快照保持独立。

## 实现

- Rust PreparedMask预编译rect/ellipse/linear/radial/brush的静态几何，polygon/raster原有预编译保留。画笔顺序、羽化、反选与覆盖数学保持一致。
- 正常图层合成跳过RGBA完全相同的像素；非normal模式继续执行混合公式。ROI外和零opacity直接保持原图。
- WASM身体前景/背景分解只保留单槽16MiB，原RGBA与target alpha精确匹配才复用；native不缓存，作为跨端参考。
- 同源图、相同上游图层/部位/蒙版的最终body.adjust连续滑杆预览保留正在运行的Worker。旧调用立即AbortError且结果不能显示，随后只算最新值，复用已完成的F/B分解。图片、上游操作、部位或蒙版变化仍终止旧CPU任务；导出快照不被预览替代。

这些优化没有修改MCP/Operation参数或引入LLM依赖。

## 验证

蒙版性质回归6项通过；四项实拍mask-only隔离回放RGBA完全一致、热中值下降23%–64%，详见 [QA](../qa/mask-efficiency-review.md)。这不是受控浏览器性能验收。

新增浏览器回归body-cache-parity.spec.ts按同source不同strength、切换部位、同尺寸不同RGB、保护变化、切回原图顺序，与无缓存native逐字节核对。重叠预览还验证旧调用取消、Worker不被无谓重建、最终结果正确。统一构建后的执行结果记录在 [current-validation](../qa/current-validation.md)。

## 边界

该缓存只复用颜色分解，不缓存新蒙版、背景补全或全部渲染。模型初始化/人体解析仍是独立成本；FASHN冷Worker约26s未达实时。大型图超16MiB上限不缓存。是否达到局部编辑1s必须在实际浏览器按分辨率测量，不能用native值代替。

## 变更记录

2026-10-13：预编译静态蒙版几何、等价合成快路径与连续身体滑杆缓存；隔离实拍效率核查与跨端/调度回归进入统一验收。
