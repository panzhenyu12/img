# V35 独立验证（2026-10-15）

结论：不建议据此替换全部瘦脸算法，也不构成 8 分验收。连续组合路径通过回归，但 portrait-woman 的脸颊目标几乎被压为零，其他强档也存在明显幅度压缩。未修改候选源码或断言。

## 固定输入与执行

已立即复制 release 二进制到 `../bin/image-render-04775f72148bec39c5994b88670d0ec6b5521c67275209cd94981a1f6b25c2b3`。构建日志确认编译的是 V35。每次渲染前后核对二进制及所有候选 Rust 源文件 hash；原图、冻结检测与输出完整记录在 `provenance.json`、`report.json`。用户图分别保留 parser 与 combined 源支持，另外两张照片使用各自原冻结检测。对照图为同一原图、同一检测与数值；15 个 parser 基线工程与新工程深比较一致，combined 5 档也一致。

三张原片、四个 source/support 组合，各 0/±30/±100，共 20 个基础输出；用户两种 support 再分别检查脸宽 30/100 加 chin/jaw 的 −.001/0/+.001，共 24 个组合输出。总计 44 次渲染，无执行错误。4 个零值输出均与原图 RGBA 一致；44 次的固定类、ROI 外、输出透明度变化均为 0。没有把相同原片的不同支持或数值计成额外原片。

## 实际幅度

| 原片/支持 | 数值 | 请求峰值 px | 投影峰值 px | 实际峰值 px |
| --- | ---: | ---: | ---: | ---: |
| user-face-contour / parser | 30 | 6.006 | 6.006 | 6.006 |
| user-face-contour / parser | 100 | 20.020 | 12.542 | 12.539 |
| user-face-contour / parser | -30 | 6.006 | 6.006 | 6.006 |
| user-face-contour / parser | -100 | 20.020 | 11.157 | 11.156 |
| user-face-contour / combined | 30 | 6.006 | 6.006 | 6.006 |
| user-face-contour / combined | 100 | 20.020 | 11.249 | 11.247 |
| user-face-contour / combined | -30 | 6.006 | 6.004 | 6.003 |
| user-face-contour / combined | -100 | 20.020 | 9.377 | 9.375 |
| portrait-woman / parser | 30 | 8.434 | 0.786 | 0.785 |
| portrait-woman / parser | 100 | 28.115 | 2.272 | 2.272 |
| portrait-woman / parser | -30 | 8.434 | 0.786 | 0.785 |
| portrait-woman / parser | -100 | 28.115 | 2.370 | 2.368 |
| portrait-makeup / parser | 30 | 7.599 | 5.522 | 5.520 |
| portrait-makeup / parser | 100 | 25.331 | 6.464 | 6.463 |
| portrait-makeup / parser | -30 | 7.599 | 5.522 | 5.521 |
| portrait-makeup / parser | -100 | 25.331 | 6.464 | 6.463 |

用户图中档约 6 px 能真实执行；+100 的实际峰值仅为请求约 20 px 的 11–13 px。颈部看起来更温和，不能排除减弱幅度的贡献，需要与旧算法相同实际移动的对照。

portrait-woman +100 请求 28.115 px，实际峰值只有 2.272 px；这项峰值还发生在归一化脸高 .90–1.12 的颈侧。脸高 .32–.66 内左/右实际峰值为 .00003/.89249 px，.66–.90 为 .00003/.65944 px。原尺寸三栏显示脸颊基本不动。候选确实启用了新曲线，因此不是未接入；目标投影已经限制了幅度。外侧 hair attachment 的限制可能参与，但本验证没有改变它，也未把这个推断当作已证明原因。

portrait-makeup +30 实际 5.520 px，+100 仅 6.463 px，明显提前饱和；原有右耳下背景重建块仍存在。用户 parser 支持中的耳环旁肤色条也仍可见。combined 支持改善了条带，但其强档移动同样压缩；不能将不同源蒙版的改善算给新曲线。

原尺寸对照在 `base/<原片>/<支持>/faceWidth_<数值>/face-comparison-1x.png`。用户额外保存 neck、leftEar、rightEar 裁切；`debug/source-width.json` 和 `debug/actual-source-width.json` 保存请求、投影与运输后曲线。

## 连续组合与回归

完整 Rust 测试：207 passed、0 failed、3 个原有手工 QA ignored；10 个公共图操作契约全部通过，见 `full-rust-regression.log`。候选现有断言未放宽或删除。

8 个跨零组合组的显式 0 都与不传该参数的结果 RGBA 完全一致。±.001 的相邻/跨零对照中，采样轮廓点最大位移差为 .000980845 px，未出现旧式路由切换导致的多像素跳跃。最终图片最多 21 个像素变化，通道差最大 2：出现在脸宽 30 加 chin +.001 的 parser/combined 两种支持，其他组最大 1。现有单测用的是更小的 ±.0001，仍保留其最大通道差 ≤1 断言；本真实照片测量没有替换或放松该断言。完整逐组结果在 `report.json.continuity`。

这些是正确性和连续性证据，不证明曲线自然或有效幅度合格。保留失败与限制，等待后继候选。渲染耗时包括 PNG 编解码和诊断输出，也处于并行开发负载下，不能直接当作正式产品性能测量。
