# 身形逆场信赖域回放（2026-11-20）

## 复现

真实照片：`test/images/fullbody-side-fashion.jpg`（1280×853）。

旧 native 回放使用 `current-native/+30.json` 时失败：

```text
body geometry folds near (703,395); requested adjustments and fixed selection/protection are incompatible
```

失败发生在最终合成前的逆位移场验证，用户滑杆没有输出图片。

## 修复后

重新编译 `core/image-core/target/release/image-render`，对同一 source、同一 graph 回放 0、−30、+30：

| 请求 | 结果 | 备注 |
| --- | --- | --- |
| 0 | 通过，输出与 source 字节一致 | `+0` identity |
| −30 | 通过 | 保护区未参与变化 |
| +30 | 通过 | 使用最大安全正 Jacobian 步长 |

输出目录（本地忽略）：`test/results/body-fresh-source-20261119/fullbody-side-fashion/current-native-next2/`。

另外在 `fashion-standing` 真实图上对腰、左右上臂、左右大腿的 0/±30/±60 共 25 个 native graph 回放全部退出码为 0。

## 限定

这次修复证明请求不再因局部网格翻折而整次失败，并保持 0 值 identity。衣物边界自然度仍需在不同衣物材质和遮挡层上独立视觉验收；本修复不把自动 `partMask` 当作光学真值。
