# FUNC-092：人脸 source-paired 公共入口接入边界

## 目的

把已经验证过的 `FaceSourceLayers` 连续材料域用于普通 `face.adjust` 之前，先明确它不能由现有图层 JSON 直接推导。当前普通入口必须保持可预测，不能为了消除接缝而把语义分割当成物理前景 alpha。

## 当前调用链

浏览器 Worker 调用 WASM `render_rgba`，Rust 进入 `apply_graph_json`、`engine::apply_graph`，再调用 `face::adjust(source, params)`。这个入口没有 `FaceSourceLayers` 参数。

连续 source-paired 算法位于独立 API：

- `adjust_face_with_source_layers(source, params, layers)`；
- `adjust_face_with_source_cache(source, params, layers, cache)`。

目前只有 native 回放例程和 Rust API 测试构造这些层，浏览器 Worker 没有准备或传输它们。

## 不能直接桥接的原因

`FaceSourceLayers` 要求同一解码源绑定的：

1. source-sized `optical_alpha: f32[]`，表示实际移动材料覆盖；
2. source-sized `scene: RgbaImage`，表示固定场景先验；
3. observation region；
4. 可选的 source motion interface 与 attachment support。

现有普通图层只包含 `subjectMask`、`semanticMasks`、landmarks 和滑杆。`subjectMask` 是人物归属概率，不能代替 `optical_alpha`；把原图直接作为 `scene` 也会把头发、耳朵、衣物或另一人当成背景供体。这样的桥接会把当前的接缝问题换成色带、旧轮廓残影或错误背景，不能进入生产入口。

## 接入前置条件

后续应增加独立的 source preparation 阶段：

1. 在客户端 Worker 或服务端生成并校验 source-bound optical alpha、foreground/background pairing 和 scene prior；
2. 以 source bytes、检测 mask、模型版本和 preparation 算法哈希作为缓存键；
3. 通过显式 `sourceSession` 或分层渲染 API 将准备结果交给 WASM/native renderer；
4. renderer 继续校验 source identity、尺寸、alpha 范围和 observation region，零请求仍逐字节 identity；
5. 准备失败时拒绝 source-paired 路径并保留手工/legacy 入口，不能静默伪造材料层。

在这些条件满足前，普通 `face.adjust` 继续使用 legacy 几何路径；复杂发丝、耳饰、眼镜和侧脸不应标记为可用。
