# 紧凑白头修复候选 v4c

本目录保留一次真实白头局部改进实验。**没有接入默认 `skin.retouch`**：油性皮肤的少数紧凑高光仍可能与白头混淆，现有传统统计无法充分区分。生产磨皮继续使用 v3 多尺度细节和红色色度场。

`skin-bright-v4c.patch` 只增补 `core/image-core/src/color.rs` 的候选 helper、调用和两个测试。候选联合检查紧凑亮度凸起、相对于周围皮肤的红绿差减弱、组件大小和有效环形 donor；修复强度随亮度与色度残差连续变化，避免二值组件边缘留下平坦圆片。它不重建囊肿几何，也不是医学分类器。

复现时，在独立的 image-core 源码副本中应用补丁，再构建 native `image-render`；不要直接在生产目录应用。固定原图、选区、参数和指标由 `scripts/check_skin_acne_quality.py` 提供：

```sh
python scripts/check_skin_acne_quality.py \
  --candidate /absolute/path/to/immutable-image-render \
  --label candidate \
  --reference /absolute/path/to/immutable-v3-image-render \
  --reference-label v3
```

原实验 binary SHA256 为 `a4490a0b5b8925cf5c99692f219d46ed741b039845c4e71df4ac49f913e034f0`。报告、已知失败和代表图见 [QA](../../doc/qa/skin-bright-v4c-review.md)。后续适合先作为“用户明确圈选的白头修复”验证，再研究像素分类与材质约束；不能因为白头一处明显变弱就默认对全脸应用。
