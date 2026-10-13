# 多样本人脸算法效果记录

日期：2026-10-13。最长边缩至768px。像素脸部控制轮廓与整头前景alpha分开；有序36点只作解剖定位。输出再检测的脸宽只作辅助测量，不能代替照片边缘视觉验收。

| 照片 | 参数 | 像素检查 | 颧侧宽度px（重新检测） | 支持区外变化 | 原轮廓外2px背景变化 |
|---|---|---|---|---|---|
| portrait-man | faceWidth=60 | 通过像素保护检查，待视觉复核 | 204.85 → 196.53 | 0 | 0 |
| portrait-woman | forehead=60 | 通过像素保护检查，待视觉复核 | 172.35 → 171.55 | 0 | 0 |
| portrait-woman | foreheadWidth=60 | 通过像素保护检查，待视觉复核 | 172.35 → 172.76 | 0 | 0 |
| portrait-woman | temples=60 | 通过像素保护检查，待视觉复核 | 172.35 → 170.33 | 0 | 0 |
| portrait-woman | cheekbones=60 | 通过像素保护检查，待视觉复核 | 172.35 → 169.61 | 0 | 0 |
| portrait-woman | jaw=60 | 通过像素保护检查，待视觉复核 | 172.35 → 172.14 | 0 | 26 |
| portrait-woman | chin=60 | 通过像素保护检查，待视觉复核 | 172.35 → 172.17 | 0 | 603 |
| portrait-woman | chinWidth=60 | 通过像素保护检查，待视觉复核 | 172.35 → 172.26 | 0 | 42 |
| portrait-woman | chinPoint=60 | 通过像素保护检查，待视觉复核 | 172.35 → 172.37 | 0 | 62 |
| portrait-woman | faceWidth=60 | FAIL | 172.35 → 167.46 | 0 | 76 |
| portrait-glasses-street | faceWidth=60 | FAIL | 114.14 → 109.14 | 0 | 2 |
| portrait-makeup | forehead=60 | 通过像素保护检查，待视觉复核 | 231.66 → 231.46 | 0 | 3 |
| portrait-makeup | foreheadWidth=60 | 通过像素保护检查，待视觉复核 | 231.66 → 232.09 | 0 | 59 |
| portrait-makeup | temples=60 | 通过像素保护检查，待视觉复核 | 231.66 → 228.57 | 0 | 3 |
| portrait-makeup | cheekbones=60 | 通过像素保护检查，待视觉复核 | 231.66 → 225.95 | 0 | 15 |
| portrait-makeup | jaw=60 | 通过像素保护检查，待视觉复核 | 231.66 → 231.18 | 0 | 175 |
| portrait-makeup | chin=60 | 通过像素保护检查，待视觉复核 | 231.66 → 231.31 | 0 | 1247 |
| portrait-makeup | chinWidth=60 | 通过像素保护检查，待视觉复核 | 231.66 → 231.73 | 0 | 147 |
| portrait-makeup | chinPoint=60 | 通过像素保护检查，待视觉复核 | 231.66 → 231.99 | 0 | 187 |
| portrait-makeup | faceWidth=60 | FAIL | 231.66 → 222.65 | 0 | 471 |
| portrait-dark | forehead=60 | FAIL | 未测得 → 未检出 | - | - |
| portrait-dark | foreheadWidth=60 | 通过像素保护检查，待视觉复核 | 170.25 → 172.48 | 0 | 0 |
| portrait-dark | temples=60 | 通过像素保护检查，待视觉复核 | 170.25 → 167.23 | 0 | 0 |
| portrait-dark | cheekbones=60 | 通过像素保护检查，待视觉复核 | 170.25 → 166.56 | 0 | 0 |
| portrait-dark | jaw=60 | 通过像素保护检查，待视觉复核 | 170.25 → 169.37 | 0 | 0 |
| portrait-dark | chin=60 | 通过像素保护检查，待视觉复核 | 170.25 → 170.04 | 0 | 631 |
| portrait-dark | chinWidth=60 | 通过像素保护检查，待视觉复核 | 170.25 → 170.43 | 0 | 119 |
| portrait-dark | chinPoint=60 | 通过像素保护检查，待视觉复核 | 170.25 → 170.31 | 0 | 0 |
| portrait-dark | faceWidth=60 | 通过像素保护检查，待视觉复核 | 170.25 → 162.60 | 0 | 0 |
| fashion-standing | faceWidth=60 | 通过像素保护检查，待视觉复核 | 84.52 → 84.13 | 0 | 0 |
| tilted-shirt | faceWidth=60 | 通过像素保护检查，待视觉复核 | 79.67 → 77.91 | 0 | 0 |
| portrait-hair | faceWidth=60 | FAIL | 172.83 → 168.83 | 0 | 16 |
| portrait-roll | forehead=60 | 通过像素保护检查，待视觉复核 | 177.75 → 177.85 | 0 | 0 |
| portrait-roll | foreheadWidth=60 | 通过像素保护检查，待视觉复核 | 177.75 → 178.68 | 0 | 47 |
| portrait-roll | temples=60 | 通过像素保护检查，待视觉复核 | 177.75 → 176.97 | 0 | 0 |
| portrait-roll | cheekbones=60 | 通过像素保护检查，待视觉复核 | 177.75 → 173.62 | 0 | 5 |
| portrait-roll | jaw=60 | 通过像素保护检查，待视觉复核 | 177.75 → 177.90 | 0 | 95 |
| portrait-roll | chin=60 | 通过像素保护检查，待视觉复核 | 177.75 → 177.67 | 0 | 757 |
| portrait-roll | chinWidth=60 | 通过像素保护检查，待视觉复核 | 177.75 → 177.40 | 0 | 86 |
| portrait-roll | chinPoint=60 | 通过像素保护检查，待视觉复核 | 177.75 → 178.52 | 0 | 101 |
| portrait-roll | faceWidth=60 | FAIL | 177.75 → 171.10 | 0 | 279 |

检测失败记录：

- clothing-texture：No face detected at evaluation resolution
- fullbody-skirt：No face detected at evaluation resolution
- fullbody-side-fashion：No face detected at evaluation resolution
- profile-hoodie：No face detected at evaluation resolution
- action-suit：No face detected at evaluation resolution

逐参数原图、输出、差分与原/输出检测轮廓保存于 test/results/portrait-algorithms。像素检查通过不代表自然度已达标。

人工目视复核（本轮40个实际输出，另1项明确拒绝）：

- portrait-makeup与portrait-roll的颧骨、下颌、下巴和脸宽能改变对应轮廓，倾斜照片沿人脸自身方向工作；眼睛、鼻子、嘴没有被统一拉伸。但强值chin等参数在耳下/颈部附近出现不自然的黄色三角填补和局部边缘接缝，不能对账为自然度验收通过。
- portrait-woman轮廓确实收缩，长发外侧大致保留，局部发丝与脸缘接缝仍需改进；faceWidth的76个保护区像素变化已列为FAIL。
- portrait-dark的faceWidth确实收缩，forehead因缺少可靠补洞供体明确拒绝。其余参数不能仅凭重新检测的脸宽没有变化就认定无效果，眉额/下巴等目标不同。
- 大幅值没有替代自然度检验。所有成功输出支持区外变化为0、四五官中心保持，只能说明已经修复整张图片/统一前景拉伸的问题，不能说明所有轮廓边界重建已完成。

这里的“原轮廓外2px”按整头前景alpha的保守扩张计算；它也可能包括颈部或衣物，报告保留原始数值，需要结合照片判断。对原始背景或邻近部位的非预期变化不按通过处理。
