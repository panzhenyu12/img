# 用户原照耳饰独立验收

首轮隔离候选未通过。8 组原尺寸复核中，+100 收脸出现耳饰旁断续肤色竖条和双轮廓，−100 扩脸局部黑楔改善但台阶仍在。保持拒绝上线，不计为 8 分。

独立核对金属高置信核心与源图完全相同，1,016 个原图中性高光像素全部保留；采样几何 SHA 相同，可信内部皮肤与 V36 字节一致，没有以降低形变幅度换取局部改善。写域外、未知域和不足补全支持区无新增修改，alpha 保持。

两种 backing 的最终 PNG 完全相同，不能声称局部背景估计已经有效修复。+100 的源支持全部拒写、移动后支持仍新增，旧边/新边不同步是下一步必须验证的问题。

详细证据：[review-v1.md](../../experiments/accessory-skin-decomposition-v1-20261015/independent-audit/review-v1.md)；独立数值：[report-v1.json](../../experiments/accessory-skin-decomposition-v1-20261015/independent-audit/report-v1.json)。生产当前版本不受本审计修改。
