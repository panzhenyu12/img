# 用户原图耳饰旁皮肤归属审计

2026-10-15，独立 source-only 观察。未改算法、模型、V36 几何、强度或服务。完整复现、坐标标签和对照图在 `experiments/user-accessory-source-ownership-20261015/`。

**v2 / hybrid 只恢复了部分真实脸颊边缘。** v2 是 pure 432 + mixed 1 = 433，总计左 264、右 169；hybrid 的右侧 81 只是更小的原组件支持子集。不能把这些计数称为轮廓已完整恢复。

在原图 1× 和最近邻 4× 中，耳饰朝向脸颊的一侧有一条连续明亮肤色边。左右各六条原图横向观察路径连接到脸颊内侧，形成两个四连通集合，共 235 个稀疏标注像素。标签只表示可见皮肤成分，不表示 alpha 必为 1。在这些少量观察里已有 10 个像素 physical alpha=0，且 v2 没有恢复。外侧混合像素与金属接触带保持 uncertain。

可复核反例包括左 `(375,450)、(376,460)、(378,470)、(379,480)`，右 `(641,450)、(640,460)、(639,470)、(638,480)`：源 RGB 可见肤色；V36 physical alpha 全为 0；旧 accessory≈1，frozen=true；v2 support=false；ZIM alpha 接近 0。它们在原图上与脸颊连续，不是根据变形结果检测出来的。

12 个边缘探针中，11 个漏选。全部在原有 6px anchor 距离内，11 个都因固定内侧 Lab 先验失败；距离不是这些点的拒绝原因。直接放宽 RGB/Lab 或把 ZIM 低值当皮肤依然不成立；应把这些探针作为源归属算法的反例与门禁输入，并保留外侧混合带的不确定性。

另一个独立问题是 source/destination 清理合同不完整：+100 即便已进入 v2 support，左 82、右 115 个源像素仍被 insufficient-scene gate 阻挡；-100 则为左 2、右 37。需要核对一条完整边缘的源清理和目标覆盖，不能仅扩大目标写入，也不能删除 refusal 便称通过。

证据文件：

- `experiments/user-accessory-source-ownership-20261015/outputs/report.json`
- `experiments/user-accessory-source-ownership-20261015/outputs/manual-observations.json`
- `experiments/user-accessory-source-ownership-20261015/outputs/component-1-sparse-labels-8x.png`
- `experiments/user-accessory-source-ownership-20261015/outputs/component-2-sparse-labels-8x.png`

本次是失败原因审计，不是修复验收。未给效果打 8 分，未部署新候选。
