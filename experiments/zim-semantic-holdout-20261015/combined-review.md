# 正确点框联合提示的独立 holdout 审计

三张独立原照：makeup、glasses-street、dark。前两张没有参与原选择规则开发；dark 是本轮对未编辑原 JPEG 的全尺寸 source-only 检测，旧 graph 未替换。每张图编码一次，所有 prompt 模式复用同一 features。仅验证单图 batch B=1。

每个自动正/负点协议都跑并保留五种调用：`upstream-mixed`、`box-only`、`points-only`、`combined-point`、`combined-box`。每种保留 4 个 raw alpha、官方分数 argmax、旧固定门槛的自动选择；没有调门槛或按视觉挑选。

`combined-point/box` 通过独立 adapter 将真实 points（含 0/1 labels）与 box corners（labels 2/3）拼接后送入 ONNX；没有额外 `-1` padding token。ORT recording proxy 记录了 50 组实际 point coords/labels、attention shape/sum/hash，全部与模式合同相符，未只相信请求参数。模型权重在加载前校验 encoder/decoder SHA。新 adapter 未修改 vendor。

## 重要纠正

旧 vendor decoder 在 `boxes is not None` 时会覆盖先前 point tokens，旧 mixed 调用实际只传 box corners；points 只影响 predictor 的 attention mask，负 label 没有到达 ONNX。因此早先旧 negative 结果不能解释成“ZIM 不尊重负点”。固定合约实验翻转实际 label 0→1，在相同合成特征下改变全部 1,048,576 logits（最大31.113、均值4.497，score最大差0.1444）；ONNX 动态点数4/7均可运行。

## 观察

- makeup 大头发：五种模式都能保留约91–93% parser hair core，skin alpha>=.5约0.24–0.53%；正/负联合模式 negative alpha 接近0。小头发组件在旧全局 hair-core>=65%门槛下全部 abstain，即使局部 mask 本身是发束；门槛限制被保留，没有为通过率放宽。
- glasses：`combined-point/box` 的自动候选均为2，hair core约91%，skin alpha>=.5约0.33–1.16%；点协议能明显压低负点，但全图仍可能有低透明幽影，二值门槛不等于物理 alpha。box-only 与 points-only 选出的候选不同，证明 attention/提示组合是实际因素。
- dark：points-only 对第二小组件可通过，combined 模式因 hair-core门槛而 abstain；大组件 combined-point hair core约95%（正点协议）、约85%（负点协议），combined-box约75%。这是固定门槛下的真实拒绝，不能为了“至少有输出”而覆盖。

所有候选的原尺寸 PNG、alpha.npy、选择决策和实际 feed metadata 在 `combined-raw/`、`combined-selection/`。这轮只证明联合调用路径工作且选择器会诚实 abstain；不证明 alpha 是物理真值，也未接入 source matte 或生产。
