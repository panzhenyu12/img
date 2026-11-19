# FUNC-091 公共脸型入口真实审计

## 目标

验证没有 source-paired 层时，普通 `face.adjust` 是否能在真实照片上完成脸宽重塑并保持背景、眼镜、头发等固定材料。

## 当前实现边界

公共入口通过 Rust `face.adjust` 调用 `adjust_impl`。它可以使用语义脸轮廓和局部 F/B 估计完成连续像素变形；没有独立前景 alpha、遮挡顺序和 source background 时，单张复合图无法可靠区分皮肤、前发、耳朵与背景。

## 验收结论

见 [face-public-legacy-replay-20261120](../qa/face-public-legacy-replay-20261120.md)。普通入口有真实轮廓响应，但发丝/耳下/滚转脸/妆容的边缘自然度仍不达标，不能标记为可用或 8/10。

## 后续实现要求

1. 检测阶段输出完整 source-bound optical alpha、固定附件和 scene donor 证据。
2. 通过 source-paired compositor 传输前景，背景使用独立 paired plate；没有这些证据时应拒绝高强度请求或提示人工修复。
3. 每项脸型、五官和妆容功能都要在至少三张真实照片上做正负方向原尺寸审图，不能把 changed-pixel 或几何 Jacobian 当成自然度通过。
