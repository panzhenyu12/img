# 人脸 source-paired 公共入口审计（2026-11-20）

## 结论

本轮没有把 `FaceSourceLayers` 强行接入普通 `face.adjust`。现有普通入口没有物理材料层输入，直接桥接会引入不可验证的前景/背景归属错误。

## 代码证据

| 检查项 | 结果 |
| --- | --- |
| WASM Worker 是否传递 source layers | 否，只传 `width/height/pixels/graphJson` |
| `engine::apply_graph` 是否持有 source layers | 否，调用 `face::adjust` |
| `FaceSourceLayers` 是否要求 source-sized optical alpha | 是，`f32[]` 且必须与 source identity 绑定 |
| `FaceSourceLayers` 是否要求独立 scene prior | 是，要求同尺寸 opaque RGBA scene |
| 普通 graph 是否包含上述两项 | 否 |
| 是否存在安全的零配置自动推导 | 否 |

## 真实照片边界

`doc/qa/face-public-legacy-replay-20261120.md` 的五张真实照片回放显示，legacy 公共入口在长发、眼镜和复杂耳下边界仍有高对比窄边或响应过弱。扩大 taper 或强制背景修补候选没有稳定改善，因此不能通过继续调几何强度替代 source pairing。

## 下一步验收

必须先完成 source preparation 和 Worker/native 双端 API，再用同一批 hair、glasses、roll、woman、makeup 原图比较 legacy 与 source-paired 输出。验收指标包括：零请求字节 identity、保护特征零变化、源轮廓外无误写、双向轮廓单调、耳下/发丝接缝无高对比色带；未满足前保持未通过状态。
