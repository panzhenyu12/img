# 用户耳饰：直接 skin 证据修正背景支持

**新观察成立：原 parser 过度保护确实使可用背景 donor 变远。** Direct full/crop skin alpha 联合 corrected accessory alpha、耳/发/五官排除和源图 RGB 四连通证据，找到 1,497 个此前被排除的已观测 scene 像素。没有因此修改成图或接生产。

左侧原 source support 到 scene 中位距离 28.12→6.71px，右侧 31.02→21px。新增域与旧 source skin support、保护前景均无交集。右侧 97 个无法连到外部 scene 的候选仍拒绝。原尺寸证据位于 `experiments/joint-skin-scene-support-v1-20261015/outputs-v2-consistent/component-{1,2}/source-evidence-1x.png`。

有效产物是 `outputs-v2-consistent/scene-support.npz`。首轮 `outputs/` 的旧 scene seed 自身有 84 个与新 direct skin/accessory 冲突的像素，已保留失败证据并剔除；不能使用首轮 prior。输出中的 `sceneDistance` 在两个局部 context 外是 inf；调用者必须检查 finite，隐藏部分 prior 仍只是假设。

完整条件、哈希、数据合同及测试见实验目录 README。此结果支持后续 source skin/scene 联合分层重建，**不代表已消除裂缝或达到 8 分**。
