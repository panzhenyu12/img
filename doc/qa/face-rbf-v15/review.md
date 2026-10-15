# 本人边缘错误冻结单因素消融 v15

与 v14 相同 RBF/目标/flow/physical-alpha/颈部代码，只纠正 source contour 定位的不确定本人边缘错误 frozen 类。输入 subjectMask 是选定人 parseFace 19类的 personAlpha，并非另一人的独立 instance。

2px RGB trimap-only 版本纠正 0 像素、SHA 与 v14 完全相同，保存在 rbf-v15-two-pixel-no-effect。本人边缘不确定域 = 现有 validate_isoline 使用的 1.5×mask texel pitch + RGB trimap 2px；1024/512 为 5px。只解除 inferred own person 边缘，显式 ears/hair/accessories/user masks 优先级不变，donor exclusion 不变。

原图 5 个右下颌残余 person 高置信像素：face/head≤.00221，physical alpha0，person .8056–.9289，neck/hair/ears/accessories0，源 contour 距离2.386–4.154px，trace 记录逐点值。+100 原点(611.165,535)以前只到(611.041,534.999)，纠正后到(591.126,534.853)，实际约20px。RMS .323→.155px。右下颌形变凹槽消失，说明 frozen 误类是这处凹槽的直接原因。

下颌/颈部黑三角和旧阴影仍失败，扩张方向 attachment target 引起圆鼓包仍失败。v15 未接生产、不是8分。保留 no-effect 对照并进入下一步：attachment target 连贯几何约束，再处理材质分层。
