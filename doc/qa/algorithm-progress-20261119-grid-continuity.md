# 脸部连续输运增量（2026-11-19）

## 本轮改动

脸部轮廓的旧实现会在轮廓附近的精确逆流与 4px 缓存网格之间做硬切换：距离小于 `3px + 4px√2` 时直接使用精确场，跨过阈值后立即改用双线性网格场。两侧场值并不保证相同，容易在轮廓接缝形成单像素位移跳变和断角。

本轮将两者放入一个有限宽度的 `smoothstep` 过渡带，过渡使用同一目标距离计算，轮廓内保持精确场，远离轮廓才完全使用网格场。叠加的脸宽、颧骨、下颌等横向请求也从硬 `clamp` 改为平滑 `tanh` 饱和，避免多个控制同时达到上限时曲率一阶导数突然归零。

## 回归

- `face::tests::contour_grid_transport_has_no_cutoff_jump`：过渡阈值两侧位移跳变小于 `0.001`，远场仍收敛到网格值。
- `face::tests::lateral_limit_is_smooth_when_multiple_face_controls_reach_the_bound`：横向目标保持单调、有限且在饱和点两侧保留正斜率。
- `face::tests::inner_face_boundary_moves_against_hair_without_moving_outer_hairstyle`：通过。
- `face::tests::each_supported_width_control_changes_its_actual_contour_in_both_directions`：通过。
- `cargo check --manifest-path core/image-core/Cargo.toml --lib`：通过；仅保留既有未使用研究模块警告。

## 验收边界

这是几何连续性修复，不等于真实照片审美通过。当前仍未达到通用可用或逐项 8 分：脸部源层/光学 alpha 仍未由 Web 公共入口稳定提供，复杂背景与遮挡需要新照片双向复核，身体自动材料归属和眼高/眼睑仍有失败边界。下一次验收必须用原尺寸多图 `0/±30/±60` 检查裂缝、曲率、五官冻结和背景写入量。

