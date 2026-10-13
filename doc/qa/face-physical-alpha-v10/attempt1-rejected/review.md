# V10 第一候选：拒绝

全局使用颜色线恢复的 physical alpha 后，已知 F/B 真值测试通过，但真实 72 项回放有 6 项失败；均为 fixture 收缩，误改同一批 5 个 head/neck 外部点，其中 4 个原 head/person alpha=0 的 scene 点。28 个 face 测试通过，full_person_matte_excludes_ears_neck_and_clothing_from_background_donors 失败，head-only 路径误改邻近人物。

该候选未合入生产。补丁、完整失败回放和测试日志在本目录；冻结可执行文件 SHA-256 为 9a46305fb4c971e37275292b898a55485b665567ec13c53004967b2d49379be8。第二候选在保留原 head 上下文保护及 head-only 输出门限后重新验证，详情见上级 review.md。
