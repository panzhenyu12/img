# 多尺度皮肤算法实拍复核

> 当前工作区源码已在本报告之后增加局部红色色度候选 v3；本页中的 v2 数值和 SHA 仍作为不可变对照保留。v3 的真实痤疮样片结果见 [skin-acne-v3-review](skin-acne-v3-review.md) 与 [skin-detail-validation-v3](skin-detail-validation-v3.json)。

日期：2026-10-13。结论：代码已改善强磨皮把正常毛孔一并抹除的问题，并让纹理保留与紧凑异常修正分别生效；当前仍未达到逐功能8分。实拍自然档改善偏弱，痣/雀斑/痘印无法可靠区分，效率有超标，不能用合同检查或模拟病例替代验收。

## 可复核版本与同参对照

基线从独立 `git archive ea5b4045a45fc5233d751c35aa945db5da7aa106` 构建，与当时已有binary逐字节校验一致，保存于 `.run/quality-baseline/image-render`。SHA256 为 `95be1d598f93cd4264bd0b439171016f57b9beb3da6b3ed2ed06611c3ca88fab`；它是旧引导滤波版，绝不是更早的Gaussian版。

本轮皮肤源码SHA256为 `949cf873f77cdb6b3b761cebc297b73635425cbed0b34120dd63ea2dc24e56c7`。独立源码快照 `.run/skin-detail-source` 的当前native binary为 `.run/skin-detail-v2-image-render`，SHA256为 `e42c9f844260d9c266c1c5875c7cb06dbb0a8779e0ba9ed15823ae1c9d212831`。快照包含采集时其他模块的状态；记录的是皮肤算法独立实验，不冒充最终全仓构建。最终共享构建的native/WASM一致性由统一验证补齐。

`scripts/check_skin_detail_quality.py` 对原缓存的同一source、蒙版、关键点和8组参数分别运行两个binary，记录图像与graph哈希。全量输出位于 `test/results/skin-detail`，逐项数据见 [validation](skin-detail-validation.json)。旧首次试验在 `test/results/skin-detail-v1`，保留 [v1 validation](skin-detail-validation-v1.json) 和 [v1 timing](skin-detail-timing-v1.json)，未删掉淡红斑修正不足的失败。

最终回放改用独立复制的不可变binary，每次spawn前后及批次结束都校验SHA256；再次生成的48张实拍和3张模拟当前输出PNG与最初v2逐字节相同，排除了可变cargo target被测试覆盖而混用版本的情况。当前全仓另建lint快照的all-targets clippy `-D warnings`、本轮Rust格式和脚本语法检查均通过。

6个版本来自5张独立原照：正面男性、深肤色男性、头发遮挡女性、带妆女性、微笑女性，以及带妆图18°旋转派生。最长边1024px。它们没有覆盖真实侧脸、极暗光、胡须、眼镜和多人；缓存只有原有skin/hair/neck及五官区域，没有新增独立accessories/ears，不能宣称这些保护完成实拍验收。

参数为 smooth60/texture60、smooth100/texture0、smooth80/texture100、whiten60、warmth±40、redness±40。本轮48项当前输出的支持域外变化0、alpha变化0；30项仅肤色输出与基线逐字节相同。没有把肤色未变的输出算作算法改善，也没有重复唇/眉/牙齿/眼下效果的旧结论。

## 实际代码改动

原算法以同一系数压低全部“原图减引导基底”的残差，smooth100/texture0会消除正常毛孔。现在保留有蒙版的512px Fast Guided Filter明暗基底，用两次归一化卷积将单像素噪声、连贯纹理及中尺度肤色起伏分别处理。基底以中尺度亮度重建，避免引导系数重新注入像素噪声；强明暗边缘降低修正，防止频率重建产生暗边/亮晕。

较暗/较红的紧凑异常单独筛选：形状、尺度、长宽比、连贯性和保护边界均参与判断，长线与大片阴影被拒绝。候选从周围干净有效皮肤估计颜色，渐变融合；texture100仍会修正这些异常，smooth0严格保持原图。正常小毛孔与低反差孤立小点不进入异常修正。检测门限的调整来自淡红斑的真实回归失败，未改动人脸/蒙版检测阈值或报告通过条件。

算法不是医学皮肤分类，也不是学习到的皮肤材质模型。痣、雀斑与痘印在经典颜色/尺度统计中可能重合；已有protectMask/skinProtection可以手工保护。凹陷疤痕、深纹与大块色素问题仍需专门算法。

## 修改前失败与已知病例

在修改算法前新增回归并实际运行：旧版强磨皮毛孔对比10→0，texture100紧凑瑕疵修正不足；第一次多尺度版本的淡红斑剩余色差15→13。新增回归不是像素变化计数，它们要求同时保留正常细节、改善已知异常，并保住宽明暗、强边缘和alpha。

| 已知病例，相同参数 | 原图 | ea5b404 | 当前 |
|---|---:|---:|---:|
| 毛孔亮度对比，smooth100/texture0 | 10.00 | 0.00 | 8.00 |
| 大瑕疵亮度对比，同上 | 39.75 | 15.30 | 10.15 |
| 毛孔亮度对比，smooth100/texture100 | 10.00 | 9.00 | 10.00 |
| 大瑕疵亮度对比，同上 | 39.75 | 37.17 | 11.72 |
| 淡红斑剩余色差，同上 | 15 | 13 | 9 |

这些是人工构造的已知场景，不能推断真实祛痘或祛斑达到8分。可看 [texture0](skin-detail-review/synthetic-texture0-comparison.png)、[texture100](skin-detail-review/synthetic-texture100-comparison.png)、[红斑四版本100%裁切](skin-detail-review/synthetic-red-patch-100pct.png)，原始数值见 [regression](skin-detail-regression.json)、[red patch](skin-detail-red-patch-regression.json)。47项颜色单测通过，包含原有保护区不能污染平滑、细噪声、强边缘、零值、透明与肤色测试。

## 实拍100%检查与评分

逐一查看六个版本的自然、强磨皮和texture100原尺寸裁切。对照图每栏都来自原尺寸图像，左原图、中ea5b404、右当前；完整支持域裁切保留五官及边界。没有只把强档改成弱参数，也没有隐藏强档失败。

| 样片 | 发现 | 证据 |
|---|---|---|
| 微笑男性fixture | 强档脸颊毛孔及笑纹细节回来，旧版整片平坦明显减轻；小斑仍有残留 | [100%](skin-detail-review/fixture-portrait-smooth-strong-100pct.png) / [完整区域](skin-detail-review/fixture-portrait-smooth-strong-comparison.png) |
| 深肤色男性 | 强档鼻侧、额头和脸颊更接近原有质感；深层细瑕疵改善仍偏保守 | [100%](skin-detail-review/portrait-dark-smooth-strong-100pct.png) / [完整区域](skin-detail-review/portrait-dark-smooth-strong-comparison.png) |
| 头发遮挡女性 | 减轻旧版无纹理的脸颊，发丝和唇齿保持；原图本身精修，不能证明祛痘能力 | [100%](skin-detail-review/portrait-hair-smooth-strong-100pct.png) |
| 带妆女性 | 恢复鼻部及脸颊细节，唇色/高光保持；原图瑕疵少，缺少有强瑕疵的正证据 | [100%](skin-detail-review/portrait-makeup-smooth-strong-100pct.png) |
| 旋转带妆派生 | 没有再出现旧版整片磨平，但不能当作真实侧脸 | [100%](skin-detail-review/portrait-roll-smooth-strong-100pct.png) |
| 微笑女性 | 下巴与脸颊不再过度平滑；浅痘印仍可见，自然档改善较弱 | [100%](skin-detail-review/portrait-woman-smooth-strong-100pct.png) / [完整区域](skin-detail-review/portrait-woman-smooth-strong-comparison.png) |

按效果35%、自然度/细节35%、部位/邻域20%、效率10%给出保守暂评。效率分用于表达已观测超标，不把受并发影响的时间包装成受控测量；完整姿态和饰物缺口使保护项暂按7评估。

| 功能 | 效果 | 自然度/细节 | 保护 | 效率 | 总分 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 自然磨皮60/60 | 6.5 | 8.5 | 7 | 5 | 7.15 | 改善保守，效果与效率维度不足7 |
| 强磨皮100/0 | 7 | 8 | 7 | 5 | 7.15 | 蜡感减轻，效率及样片覆盖不足 |
| 纹理保留80/100 | 7.5 | 8.5 | 7 | 5 | 7.50 | 已知瑕疵修正独立，实拍分类仍不足 |

分数是目前照片的主观暂评；不能拿已知病例的改善量换算视觉评分。肤色算法和结果逐字节未变，仍沿用旧报告指出的冷调偏粉紫、美白局部层次不足，不能说本轮修好了。唇、眉、牙齿、眼下不是本轮验证范围。

## 效率与剩余失败

最新版48项native总进程463.13–5307.76ms，包含启动、PNG解码、graph解析/验证、编辑、合成和PNG编码，不是热编辑时间。新增 `examples/skin_quality_bench.rs` 在同一进程里把解码移出计时，分别记录第一次及5次热运行；仍包含graph解析/验证、编辑及最终合成，未调用模型，也不能填写检测或模型初始化时间。最新12个强磨皮/美白病例的热中值555.28–6093.67ms，详见 [timing](skin-detail-timing.json)。

现场有同时回放/benchmark、编译和Chrome高CPU负载，时间不构成受控性能验收；原始数据全部保留，部分已经超过目标。1024px简单肤色200ms及局部传统编辑1s目标均未全面满足，不称实时。本轮没有测峰值内存或浏览器WASM热帧时间。首轮24个病例的热中值320.08–4101.74ms也保留，不能挑最快样本替代中值。

下一步需单独优化及测量graph/蒙版处理、核心编辑和浏览器显示各自耗时；补有明显痘印、痣/雀斑、真实侧脸、胡须/眼镜和暗光的独立样片，避免仅在五张已有原照上调启发式。眼袋/泪沟结构、自动妆容材质及衣物重建仍不是本算法完成项。
