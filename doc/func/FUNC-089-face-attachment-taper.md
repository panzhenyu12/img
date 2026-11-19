# FUNC-089 脸部固定材质接缝运动过渡

## 目的

脸宽、下颌等连续形变在发丝、耳朵或用户保护区旁边进入固定材质时，不能把位移集中到一两行像素。原实现的 source paired 路径使用较短的运动域 taper，虽然场的 Jacobian 仍为正，但成片会出现近垂直平台或小回折。

## 实现

`core/image-core/src/face.rs` 的 source paired 路径使用 `motion_domain_taper`：

- source paired：`clamp(frame_width × 0.16, 10, 36)`；
- legacy：保留 `clamp(frame_width × 0.12, 6, 24)`。

该参数只控制固定材质边界附近的运动增益。材料归属、alpha、donor ownership、保护像素和脸宽目标不变；脸颊内部在 taper 外仍为完整请求位移。通过更宽的 C2 motion-domain ramp 把接缝位移分散到更多轮廓采样，避免以降低全脸强度换取平滑。

## 验证

- 单测：`source_paired_motion_domain_uses_a_wider_attachment_taper` 通过；legacy 与 paired 的上下界和宽度均锁定。
- 单测：`fixed_attachment_transition_spreads_motion_without_a_platform` 通过。
- fresh hair 真实源回放：`/tmp/face-replay-20261119-face-taper-profile-2/`，784×1024，`faceWidth=+60/-60`，同源 graph；两档均保留脸部响应、显式保护变化为 0、声明区域外变化为 0。
- 与旧 fresh-source 回放相比，+60/-60 的变更像素分别由 34915/35365 降至 27593/27631，最大通道差由 184/208 降至 156/180。该指标只说明边界写入减少，仍需原尺寸人工审图，不能单独视为通用自然度通过。

## 限制

该修复只处理固定材质接缝的运动过渡。它不补齐错误的 hair/garment instance identity，也不替代真实 alpha 和源层审查；其他姿态、五官和身形功能必须分别回放验收。
