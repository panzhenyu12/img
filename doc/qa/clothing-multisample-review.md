# 衣物局部算法真实样本记录

日期：2026-10-13。三个手工衣物内部选区，最长边1024px。只检查局部调色、细褶纹理弱化与纹理保留；这组能力不能宣称几何去褶皱。纹理高/低均固定smooth=80；smooth=0时纹理保留滑杆应保持原图。高频梯度降低不能代表大褶皱消失，仍需视觉检查。

| 照片 | 参数 | 选区内变化 | 选区外变化 | 梯度能量 |
|---|---|---|---|---|
| clothing-texture | smooth | 19338 | 0 | 3.041 |
| clothing-texture | texture-low | 26084 | 0 | 1.646 |
| clothing-texture | texture-high | 19883 | 0 | 3.103 |
| clothing-texture | hue | 29315 | 0 | 4.081 |
| clothing-texture | saturation | 29315 | 0 | 4.089 |
| clothing-texture | brightness | 29315 | 0 | 3.739 |
| clothing-texture | texture-without-smooth | 0 | 0 | 4.080 |
| tilted-shirt | smooth | 8824 | 0 | 5.216 |
| tilted-shirt | texture-low | 10799 | 0 | 2.682 |
| tilted-shirt | texture-high | 8946 | 0 | 5.335 |
| tilted-shirt | hue | 11152 | 0 | 7.245 |
| tilted-shirt | saturation | 11152 | 0 | 7.184 |
| tilted-shirt | brightness | 11152 | 0 | 7.163 |
| tilted-shirt | texture-without-smooth | 0 | 0 | 7.158 |
| fashion-standing | smooth | 13580 | 0 | 5.531 |
| fashion-standing | texture-low | 14645 | 0 | 2.644 |
| fashion-standing | texture-high | 13737 | 0 | 5.676 |
| fashion-standing | hue | 14760 | 0 | 7.837 |
| fashion-standing | saturation | 14759 | 0 | 7.851 |
| fashion-standing | brightness | 14760 | 0 | 7.719 |
| fashion-standing | texture-without-smooth | 0 | 0 | 7.720 |

原图/输出/差分图保存于test/results/clothing-algorithms。灰白衣物缺少彩色像素时色相响应可能很小，不能单靠changed断言效果好坏。

人工目视结论（本轮样本）：

- clothing-texture：织物细纹和小折痕更柔和，texture=0比100明显更模糊；大块褶皱阴影仍在。
- tilted-shirt：袖子纹理与细斜褶有弱化，袖子主要折线和轮廓保留，未完成几何熨平。
- fashion-standing：细密织纹有平滑，较高texture能保留更多细节；钮扣和大折影仍存在。

因此当前功能名称为“褶皱细纹弱化”，不能对账为已完成真实衣物去褶皱。
