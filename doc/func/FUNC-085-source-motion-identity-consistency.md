# FUNC-085：源光学层与形变材料身份一致性

2026-11-19。独立算法修复，未接 Web/MCP。侧身照片在人工源纠正后的 `0/±30` 有限通过；不代表自动分层成功或全部身形功能可用。

## 问题与修改

静止外套在光学源分层中已正确固定，旧 `partMask` 却仍把它算成腰部。右侧轮廓测量由外套宽度骤变到衣身宽度，`+30` 被实际 Jacobian 检查拒绝。源图重建精确、alpha 合法，均不能证明运动身份正确。修复不减小强度、不取消翻折检查，也没有增加形变场补丁。

`experiments/body-material-contrastive-20261119/geometry.py` 将明确的原图运动角色统一用于可见轮廓测量：moving-rear 与原主体观测相交，排除 stationary-front、stationary-scene 和明确替换角色的源提案。后层被遮住部分的估计不能成为可见轮廓。前层提案内部的小封闭遗漏按身份未决排除，不填充物理 alpha，也不修改原图。

替换提案按显式 role ID 绑定，不靠照片名称或固定衣袖名字分支。角色必须唯一且存在，提案是源尺寸布尔分类；连续 alpha、维度错误、未知角色均拒绝。固定角色优先于后层完整支持。

曾尝试叠加 FASHN 衣物分类过滤，结果误删粉色针织翻折和裙上字带。该 V2 被拒绝。现在粗分类仅报告冲突供检查，不覆盖明确的人工源身份。V3/V4 恢复被误删的 3,810 个像素，最终 74,472 个 moving 像素与 V1 提案逐字节相同。

## 来源与输入成本

同目录 `probe.py` 用两正两负手工点的相反查询检查 SAM2 部件候选；`prompted.py` 提供源图、手工请求与既定审图范围的普通实验入口。每次只编码一次，保留每个查询的全部三候选，不按分数挑最好。side-fashion 的一个袖提案可用，tilted-shirt 六候选均未完整区分袖子与胸衣。

`source_layers.py` 从袖提案重做前层 MEMatte、隐藏底图、后层 MEMatte 与场景补全。其他手、包、外套和后层身份仍来自显式手工标注；不能声称四个点完成整人自动分层。所有源层先于位移准备，源重合成误差约 `2.84e-14`，但这些仍是单图估计。

## 实际验收

最终回放复用同一组五个光学源层，仅纠正运动部位和重测支持域，保留原 Pose、保护和请求强度。零值逐 RGBA 相同；`−30/+30` 分别改变 6,739/6,374 像素，不透明前层、原保护和零场像素均不变。三档均通过原几何检查。

独立审图查看全图、完整腰部、前袖、外套及恢复针织的接界。左侧可见裙腰双向响应成立；未见新增裂缝、双纹和暗线，袖、手包及外套保持稳定。右腰被遮挡，不能声称验证了隐藏腰宽。肩部提案窄尾和下摆边界不能推广为肩部/腿部运动真值。详见 [独立审查](../qa/body-source-geometry-independent-20261119.md)。

4 项几何身份行为测试通过；提示入口 5 项、源 trimap 2 项测试只验证各自输入和不确定带行为，不替代照片验收。数据、模型、标签和成片仅保留本地。

## 复现

运行目录与输入均为本地原图证据；新输出目录不得存在：

```sh
services/background-worker/.venv/bin/python experiments/body-material-contrastive-20261119/geometry.py \
  test/results/body-material-candidates-20261119/side-fashion-source \
  test/results/body-fresh-source-20261119/fullbody-side-fashion/manual-role-control-v3 \
  test/results/body-material-contrastive-20261119/source-layers \
  /tmp/side-visible-geometry-new --replacement-role sweater-front-arm

services/background-worker/.venv/bin/python -m unittest discover \
  -s experiments/body-material-contrastive-20261119 -p test_geometry.py
```

最终源几何为 `visible-source-geometry-v4`，mask SHA-256 为 `d14a747ff1662f53fb95c1a538b20d8971b34d64f1a68c6953df7983296aa7ca`。已审成片位于 `source-layers/geometry-corrected-replay-v3`，其完整命令及输入哈希记录在 `report.json`。V4 仅移除替换角色的样本专属代码，输出 mask 不变，因此没有重复出图。旧 geometry-corrected-replay 对应被拒绝的 V2 分类输入，保留用于对照。
