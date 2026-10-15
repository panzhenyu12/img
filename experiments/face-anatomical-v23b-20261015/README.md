# v23b 颈部 donor 桥接消融（拒绝）

以 v23 的 join 后脸部流场归零为基线，唯一改动是让 joint-neck 路径调用已有 `restore_neck_lighting`。目的是验证黑色露底能否由旧颈部 donor bridge 修复。原图五档已渲染到 `outputs/full`；+100 仍有黑楔、颈侧色块和硬折，-100 仍有连接断点。说明旧 bridge 依赖旧脸部目标，不能直接接在新的 zero-flow join 后。候选拒绝，未改 production。
