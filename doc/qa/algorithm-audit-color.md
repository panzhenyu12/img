# 颜色、皮肤、衣物与胶片算法对账

日期：2026-10-13。实现：`core/image-core/src/color.rs`。

本次对账以参数方向、已知数学输入、目标部位、冻结保护、取样来源和透明度作为验收条件。旧的 `parameter-pixel-checks.json` 只记录变化像素数，不能作为算法效果合格的依据。当前实现属于可验证的经典算法 demo，尚无世界领先质量的证据。

## 已确认并修正的缺陷

| 缺陷 | 修正前实测或代码证据 | 修正 |
| --- | --- | --- |
| 磨皮从手工选区外取色 | 96×96 图内选中的皮肤完全相同，仅更换选区外颜色，选区内 616 像素变化，最大通道差 8 | RGB 与有效取样权重同时卷积，再归一化；donor 包含手工 mask，排除五官、冻结像素及透明像素 |
| 衣物平滑把冻结对象颜色吸入衣物 | 左半衣物不变，右半冻结区域改红；衣物内 2112 像素变化，最大通道差 59，边缘 `[180,140,110]` 变成 `[212,81,64]` | 衣物在 region/mask 内取样，冻结区不参与卷积；同样使用归一化卷积 |
| 衣物平滑强度从 0 到 1% 跳变 | 非零 smooth 立即乘与 smooth 无关的纹理衰减系数 | 强度从 identity 连续插值，增加 1%/50%/100% 误差单调性测试 |
| 曝光在 sRGB 编码值上直接乘倍数 | +1 EV 会把 128 接近推到 255，失去光强翻倍含义 | 在线性光中乘 `2^EV`，再转回 sRGB；+1 EV 的参考结果为 64→约90、128→约176 |
| 黑白滑杆不是渐进效果 | 任意 `blackWhite > 0` 立即全黑白 | 0–100% 在原色和灰度之间连续混合，并保护亮度 |
| 清晰度、纹理、锐化只是同一 residual 的不同系数 | 共享同一个 Gaussian 高通 | 改为独立大、中、小空间尺度；尺度随图像尺寸调整 |
| 降噪直接 Gaussian，损坏轮廓 | 与锐化共用低频图并相减 | 改为 bilateral 空间/颜色双核，在 25 邻域内减少小噪声并保留强边缘 |
| 去雾实际是对比度和压暗 | 没有大气光和透射率 | 改为 dark channel prior、大气光估计、最小值滤波、亮度引导的局部线性透射率细化和线性光重建 |
| 去雾 native/WASM 偶发差1字节 | 128×128 周期彩色 grid、dehaze=60，(96,121) 红通道 native=74、WASM=75；平台 powf 与动态 Gaussian exp 核会放大半字节取整差异 | sRGB、曝光和gamma使用无架构优化的libm软件幂函数；去雾改成固定遍历、f64累积、排除无效donor的加权box guided filter，不放宽输出误差 |
| 胶片 master 强度为 0 时特效仍生效 | grain/fade/vignette/glow 在 preset 强度之后计算 | 最后统一混合所有效果；强度 0 必须逐字节原图，50% 必须为近似中间值 |
| halation 只给亮部加红 | 未区分高光核心与外部晕边 | 对高光发光 mask 做空间扩散，减去高光核心，仅在外缘形成红色晕边 |
| HSL 红色色阶会改变纯灰的亮度 | 灰色 HSL hue 返回 0，被当成红色 | 色阶选取乘色度权重，无定义色相的灰色不参与指定色相调整 |
| 眼袋/泪沟被当作提亮 | 与黑眼圈共用一个 brightness 公式 | 移除眼袋/泪沟的颜色伪实现；协议拒绝非零值，UI 暂停显示；黑眼圈保留局部色调校正 |

## 每个公开颜色参数的效果合同

表内“通过”表示以下性质测试通过，不表示真实摄影全场景质量合格。

### 基础调色 `color.basic`

每项均额外验证：选区外和冻结像素逐字节不变、alpha 不变、opacity=0 为原图。

| 参数 | 独立效果合同 | 结果 |
| --- | --- | --- |
| exposure | +1 EV 符合线性光 sRGB 参考值；负 EV 变暗；纯黑仍黑 | 通过 |
| brightness | 正值等量提亮、负值压暗，在两种灰度上方向一致 | 通过 |
| contrast | 正值使低于中灰更暗、高于中灰更亮；负值相反 | 通过 |
| highlights | 正负方向正确，对亮部影响大于暗部 | 通过 |
| shadows | 正负方向正确，对暗部影响大于亮部 | 通过 |
| whites | 正负方向正确，较高光控制更集中于亮端 | 通过 |
| blacks | 正负方向正确，集中于暗端 | 通过 |
| temperature | 正值红相对蓝增加，负值蓝相对红增加 | 通过 |
| tint | 正值向洋红、负值向绿色 | 通过 |
| saturation | 正值扩大色度；-100% 灰度化；灰色不染色 | 通过 |
| vibrance | 低饱和色的相对增益大于已饱和色；灰色不染色 | 通过 |
| clarity | 增强较粗尺度的局部反差；负值削弱反差；响应与锐化不同 | 通过 |
| sharpness | 增强细尺度边缘/细节，响应与纹理、清晰度不同 | 通过 |
| texture | 正值增加细节，负值减少细节；独立空间尺度 | 通过 |
| noise | 棋盘小噪声差由 8 降到 <4，跨主体边缘灰度差仍 >125 | 通过 |
| dehaze | 已知 `t=0.55` 的大气散射样片，复原误差低于雾图一半；负值加雾；无可估计结构的均匀色块保持 | 通过 |
| blackWhite | 25%/50%/100% 色度递减，100% 完全灰度，亮度保持 | 通过 |
| blackPoint | 设定黑端附近归零，中间调压暗；输出单调 | 通过 |
| whitePoint | 设定白端以上归白，中间调提亮；输出单调 | 通过 |
| gamma | >1 提亮中间调，<1 压暗；0/255 端点不动；输出单调 | 通过 |

相关测试：`basic_neutral_settings_preserve_every_channel_and_alpha`、`exposure_obeys_linear_light_ev_instead_of_encoded_byte_multiplication`、`tonal_sliders_have_correct_direction_and_light_range_preference`、`levels_and_gamma_have_stable_endpoints_and_monotone_midtone_behavior`、`temperature_tint_saturation_and_vibrance_match_their_color_axes`、`black_white_is_a_progressive_mix_and_preserves_luminance`、`detail_controls_use_distinct_scales_and_signed_strengths`、`bilateral_noise_reduction_suppresses_noise_without_smearing_a_silhouette`、`dark_channel_dehaze_reduces_error_on_a_known_atmospheric_veil`、`every_public_basic_parameter_obeys_mask_freeze_opacity_and_alpha`。

### 皮肤 `skin.retouch` 与妆容颜色 `feature.adjust`

| 参数 | 独立效果合同 | 结果 |
| --- | --- | --- |
| smooth | 减少细纹/小噪声的局部色差；五官、背景与显式 hair/skinProtection 不变；禁止无效 donor | 通过 |
| texture | smooth=0 时任何纹理参数均不改图；smooth>0 时高纹理保留比低纹理保留更多细节 | 通过 |
| whiten | 提高皮肤亮度，保持正常肤色的通道差；保护深阴影和高光、五官 | 通过 |
| warmth | 冷暖改变色度、亮度误差 <1 字节；正值偏暖、负值偏冷 | 通过 |
| redness | 红/绿差按参数方向变化，亮度误差 <1 字节 | 通过 |
| browColor | 正值加深眉毛；眉毛语义区之外、冻结眉毛不变 | 通过 |
| lipColor | 增加唇部红色偏向；排除内嘴/牙齿；冻结唇区不变 | 通过 |
| lipBrightness | 提高唇部亮度，内嘴/牙齿不染色，冻结唇区不变 | 通过 |
| lipSaturation | 增加唇部色度，内嘴/牙齿不染色，冻结唇区不变 | 通过 |
| teethWhiten | 提亮低饱和高亮牙齿并减少黄色；红唇、暗口腔不变 | 通过 |
| darkCircle | 仅在眼下语义区提升亮度；眼睛/眉毛与冻结区不变 | 通过；只是色调校正 |
| eyeBag | 禁止用提亮冒充结构修复 | 未实现；非零值拒绝 |
| tearTrough | 禁止用提亮冒充结构修复 | 未实现；非零值拒绝 |

相关测试：`skin_redness_is_chromatic_and_texture_preserves_detail_only_when_smoothing`、`every_makeup_color_parameter_has_its_own_direction_and_semantic_boundary`、`skin_tone_separates_luminance_and_chroma_and_protects_highlights`、`skin_color_preserves_deep_shadows_and_specular_highlights`、`semantic_skin_mask_protects_features_and_background`、`skin_retouch_preserves_explicit_hair_and_skin_protection`、`skin_smoothing_cannot_import_color_from_outside_the_selection`、`protected_feature_colors_cannot_contaminate_neighboring_skin_smoothing`、`lip_adjustments_leave_inner_mouth_and_teeth_untinted`、`teeth_whitening_runs_without_lip_mask_and_rejects_red_and_dark_pixels`、`under_eye_controls_protect_eyes_and_brows_even_when_masks_overlap`、`structural_eye_bag_and_tear_trough_are_not_faked_with_brightness`。

### 衣物颜色与局部平滑 `clothing.adjust`

| 参数 | 独立效果合同 | 结果 |
| --- | --- | --- |
| smooth | 强度从 0 连续增加；选中纹理仍被有效平滑；未选衣物/冻结皮肤不参与取样 | 通过；不等同复杂褶皱重建 |
| texture | smooth=0 时保持原图；smooth>0 时高值保留更多织物细节 | 通过 |
| hue | 50% 对应 +90° 色相旋转，仅改变选中衣物 | 通过 |
| saturation | 增加选中衣物色度，其他部位与 alpha 不变 | 通过 |
| brightness | 提亮选中衣物，其他部位与 alpha 不变 | 通过 |

相关测试：`clothing_hue_saturation_brightness_and_texture_have_correct_local_semantics`、`clothing_smoothing_reduces_fold_contrast_and_keeps_fabric_detail`、`clothing_smoothing_strength_is_continuous_and_monotone_from_zero`、`clothing_smoothing_excludes_unselected_and_frozen_color_donors`。

### 曲线、HSL 与 LUT

| 操作/参数 | 独立效果合同 | 结果 |
| --- | --- | --- |
| curves.points | identity 精确不变；排序后线性插值，128 附近映射到192；重复 x 只有一个有效位置时拒绝 | 通过 |
| curves.channel | red/r、green/g、blue/b 仅改变指定通道；rgb/master 改三通道；alpha 不变 | 通过 |
| hsl.hue | 100% 全局色相转180°；灰色保持灰色 | 通过 |
| hsl.saturation | 正值增加色度，-100% 变灰 | 通过 |
| hsl.lightness | 正值提亮 | 通过 |
| hsl.hsl/channels | 两种参数入口结果一致；按原始 hue 选择，指定红色时蓝色和纯灰不变 | 通过 |
| hsl 色阶 hue/range | 色相选择和圆周距离有效，窄色阶排除区域外色相 | 通过 |
| hsl 色阶 shift/saturation/lightness | 分别旋转色相、增加色度、提亮；其他色阶不动 | 通过 |
| lut.size/data | 3D 三线性插值，RGB 轴/序列顺序正确；2³ 通道交换 LUT 精确交换 R/B；错误点数拒绝 | 通过 |
| lut.amount | 0% 为原图；50% 为数学中间值 | 通过 |
| lut.domainMin/domainMax | 各通道独立映射输入域，输入端点按预期夹紧；无效域拒绝 | 通过 |

相关测试：`curves_identity_and_lut_identity_preserve_alpha`、`curves_interpolate_exactly_and_only_touch_the_selected_channel`、`hsl_global_and_selective_adjustments_respect_hue_saturation_lightness`、`hsl_changes_selected_hue_without_changing_alpha`、`lut_trilinear_interpolation_channel_order_amount_and_domain_are_correct`、`lut_rejects_bad_point_count`。

### 胶片 `filter.film`

| 参数 | 独立效果合同 | 结果 |
| --- | --- | --- |
| intensity | 0% 所有效果关闭，50% 为最终输出中间值 | 通过 |
| grain | 固定 seed 可复现，换 seed 改纹理；噪声均值近0；RGB 同幅变化，不产生彩色偏移 | 通过 |
| seed | 相同 seed/输入稳定、不同 seed 纹理不同 | 通过 |
| vignette | 周边比中心暗，alpha 不变 | 通过 |
| glow | 扩散高光，邻近亮区出现无偏色亮光 | 通过 |
| halation | 高光外缘出现红色晕边，核心与远处暗区不变 | 通过 |
| fade | 黑位抬高、白位降低 | 通过 |
| preset | neutral identity；warm 偏暖；cool 偏冷；classic 降低高饱和色并加暖；fade 抬黑；vivid 增色度；mono 灰度；cinema 暗部偏青 | 通过；手工风格公式 |

相关测试：`film_master_intensity_scales_every_effect_and_zero_is_identity`、`film_grain_is_repeatable_zero_mean_and_seeded_without_color_drift`、`film_vignette_fade_glow_and_halation_have_distinct_spatial_effects`、`each_film_preset_has_its_declared_color_direction`、`spatial_color_algorithms_cannot_import_frozen_or_unselected_colors`。

## 当前验证

```bash
cd core/image-core
cargo test --lib color::tests -- --nocapture
cargo clippy --lib -- -D warnings
```

颜色模块 42 项单测通过。包含逐参数效果合同、已知去雾场景复原误差、独立空间尺度、保边降噪、取样隔离与保护回归；新增软件 sRGB 全256编码值精确往返和加权guided统计的已知均值/排除donor测试。完整工程、WASM/native 一致性、真实照片视觉回归以统一验收报告为准。

最终去雾修复已执行 `make native wasm` 重建两端。`apps/web/tests/dehaze-parity.spec.ts` 显式比较解码后的全部RGBA字节，覆盖原失败128×128 grid、dehaze=60，另覆盖负60，以及271×139分析缩放+冻结蒙版的100强度；三场景差异均严格为0，Playwright通过，算法相关测试执行约4.1秒。没有设置1字节容差。新helper回归同时验证sRGB转移单调且精确往返、常量场统计保持、已知ramp均值正确，以及不合法donor即使改成极大值也不能污染统计。该证据证明这些输入的跨端一致性，不证明真实雾景自然度。

## 距离专业/世界领先质量仍缺什么

1. 色彩管理：当前按 sRGB 假设处理，缺少 ICC/广色域、RAW、高精度浮点工作空间、可靠 HDR 和专业色相/明度空间。线性曝光已修正，其他调色参数仍是简化色调算法。
2. 真实皮肤解析：MediaPipe 面部凸包与五官轮廓不是像素级 skin/hair/beard 分割。显式保护 mask 的合同已通过，但自动检测尚不能在头发遮脸、胡须、眼镜、遮挡等样片上证明安全。
3. 皮肤与妆容：磨皮仍是保边残差抑制；唇色为确定性色度偏移，不是妆容材质迁移。黑眼圈只是局部色调减弱，眼袋和泪沟没有结构/光照重建算法。
4. 衣服去褶皱：当前局部平滑降低纹理/光影起伏，不能独立识别折痕和织物细纹，也不能重建复杂服装几何、缝线与阴影；不应宣称论文级去褶皱完成。
5. 细节工具：多尺度 residual 与 bilateral 是有明确性质的经典方法，仍缺少锐化晕边抑制、多尺度自适应纹理分析、噪声估计、毛孔/织物辨别与真实高 ISO 基准。
6. 去雾：dark channel prior 在白墙、雪、天空和无暗色物体的画面上可能误估大气光；当前采用256分析图、灰度引导的加权box统计和透射率上采样，并非原始DCP论文的完整soft matting或全分辨率计算。合成场景误差测试通过只验证物理方向，缺少真实雾景benchmark。
7. 胶片：现有 preset 为手工色调风格，未使用富士官方模拟数据或校准真实胶片响应；grain/halation 为图像效果近似，缺少颗粒尺寸随输出分辨率变化、密度依赖、镜头/乳剂颜色与实拍标定。不能称富士复刻或物理胶片模拟。
8. 质量评估：需要经授权的多肤色、多姿态、遮挡、光照、分辨率、噪声与复杂衣物样片集；人工盲评和明确失败门槛，结合 PSNR/SSIM/色差/轮廓与边缘指标。单测成功不代表自然度达到专业或世界领先。

算法依据：Dark Channel Prior（He, Sun, Tang, CVPR 2009）、Guided Image Filtering 的局部线性思想（He, Sun, Tang, ECCV 2010）、Bilateral Filtering（Tomasi, Manduchi, ICCV 1998）。当前代码采用有明确标注的简化实现，没有把论文名当作已完整复现的证明。
