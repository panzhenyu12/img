# FUNC-006 图形变换、网格与液化

状态：已验证（2026-10-12）

关联需求：WARP-001、WARP-003、WARP-004、WARP-005、WARP-008、WARP-009（图像尺寸）、WARP-010、WARP-013、WARP-014；WARP-002双向twirl、WARP-006液化重建笔刷、WARP-007画布中线对称液化。

## 用户行为与实现

用户输入裁剪矩形/旋转角/目标尺寸，或拖动3×3网格和四角透视。液化提供推移、收缩、膨胀、双向旋转、重建和对称镜像；冻结保护保留背景，节点可关闭、撤销或删除。前端只把手势转成params/controlPoints，Rust geometry.rs负责反向采样；native使用同一算法。

## Operation 与参数

```json
{"type":"crop","params":{"x":10,"y":20,"width":800,"height":600}}
{"type":"rotate","params":{"degrees":12.5,"expand":true}}
{"type":"flip","params":{"axis":"horizontal"}}
{"type":"resize","params":{"width":1200,"height":800}}
{"type":"perspective","params":{"corners":[{"x":0.1,"y":0.1},{"x":1,"y":0},{"x":1,"y":1},{"x":0,"y":1}]}}
{"type":"warp.grid","params":{},"controlPoints":[{"x":0.5,"y":0.5,"inX":0.5,"inY":0.5,"outX":0.6,"outY":0.5}]}
{"type":"warp.liquify","params":{"strokes":[{"x":0.5,"y":0.5,"toX":0.6,"toY":0.5,"radius":0.1,"strength":60,"mode":"push"}]}}
```

画布尺寸发生改变时不允许局部蒙版和不透明度混合；UI隐藏该几何节点的这些选项，协议层错误明确返回。

## 完成标准与测试

Rust测试精确crop/flip/90°像素坐标、任意旋转透明角、透视identity和退化四边形；Go同一真人照片逐类执行；Playwright拖动网格/透视点、液化笔划和保护蒙版，确认像素变化，并验证裁剪、尺寸与JPEG输出。

## 当前限制与变更

网格为加权逆向位移，未接rust_mls/Delaunay。重建恢复本液化层输入，保留更早操作；对称twirl镜像方向相反。当前无独立画布扩展和镜头校正。滑杆从原图与序列重算，不在预览上累计；液化内部回放多笔划仍有重复采样损失。2026-10-12 统一至shared Rust，加入任意旋转、resize、透视、网格、液化和保护。
