# 旧代码可复现的回归失败

- `exposed_cheek_cannot_reenter_original_opaque_skin_after_a_background_gap`：旧版失败，output `(106.5,160.5)` 映射到 source `(100.59534,162.44424)`，在目标外采到不透明皮肤；候选通过。
- `stationary_ear_attachment_preserves_the_physical_two_pixel_trimap`：旧版失败，目标根部不动仍出现非零内部位移；候选通过。
- `fixed_ear_attachment_contours_remain_connected_in_both_directions`：确认双向耳根目标恒等，无遮挡下脸保留完整请求量。

三项只是几何行为回归，不代表真实照片质量达标。旧版源来自 `source-provenance.json` 的冻结 face.rs；失败测试在 `.run/face-side-v12-baseline-test-core`，没有改共享源码。
