# 独立外颈流场 v23

以v22源弧比例/可编辑颊弧为冻结基线，只改外颈流场的连接检测和局部权重。使用物理source外边界，过滤过高的脸部join证据；真实颈join从frame下方约0.62h开始搜索。脸部velocity在join后按局部有界权重快速归零，保留脸颊20px请求，不全局削弱faceWidth。保留v22失败图与support/ownership已知问题；不改production。

## 失败记录

v23 原尺寸回放在 +100/-100 的脸颈交界产生黑色凹口和硬折线。`joint_neck` 路径当时跳过 `restore_neck_lighting`，但仅恢复该调用仍会留下棕色色块（v23b），因此该方案拒绝，未集成 production。
