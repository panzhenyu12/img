# 身形光学材料域回归（2026-11-19）

## 变更

`body.adjust` 增加可选 `opticalAlpha`。几何目标继续负责运动，光学 alpha 负责材料输运与合成，避免宽 `subjectMask` 把没有被选中衣物的像素带入前景。

## 证据

| 检查 | 结果 |
|---|---|
| `opticalAlpha` 超出 `subjectMask` | 明确拒绝 |
| 独立材料 alpha 的非材料目标区 | RGBA 字节保持不变 |
| 独立材料 alpha 的移动材料 | 有非零响应 |
| 对角 opaque 目标合成 | 与逆场源采样最大通道误差 ≤ 1 |
| body 单元测试 | 28 passed，1 ignored |
| validation 单元测试 | 8 passed |
| image-core 全量 lib | 507 passed，9 ignored |

## 真实照片状态

`fashion-standing.jpg` 的自动衣物实例仍有源身份错误：独立外套和袖子无法仅靠现有语义 proposal 自动分离。该回归只证明材料域接口和合成边界，不把自动服饰分层计入可用性评分。下一步需要在固定 source identity 后验证原尺寸 `0/±30`，并保留失败样本。
