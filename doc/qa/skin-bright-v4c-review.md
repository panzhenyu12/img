# 白头局部修复 v4c：保留实验，不进入默认磨皮

日期：2026-10-13。三张真实痤疮原照、固定手工 skin polygon、natural/strong/texture 共九项；使用与 v3 完全相同的输入、选区和参数。候选确实减弱了标注白头，但对少数紧凑油脂高光仍有歧义，因此生产 `skin.retouch` 已恢复 v3。没有给予 8/10，也不将囊肿没有改善计为成功。

## 实验与可复现性

候选新增亮度凸起、相对周围皮肤减色度、紧凑组件和有效环形 donor 检查；连续亮度/色度权重替代二值核心替换，避免第一版的平坦硬圆片。同色加性照明、低对比毛孔、宽近白高光、无有效 donor 和冻结区均有合成负例。

候选 binary：`.run/skin-bright-v4c-image-render`，SHA256 `a4490a0b5b8925cf5c99692f219d46ed741b039845c4e71df4ac49f913e034f0`。参考 v3 SHA256 `8aacc21470c2262fb9dfbe1f716bd1f30197fcfcc19c1cb8e569df8cd00751db`。逐项输入/原照/graph 哈希、参数和未通过项见 [九项原始指标](skin-acne-v4c.json)。实验源码补丁与复现说明见 [实验目录](../../experiments/skin-bright-spots/README.md)。

候选 50 项颜色 release tests 通过，包含两个新增白头/高光/保护测试；50 是独立候选的数量，不是默认生产的测试数量。生产恢复后 `clippy --all-targets -- -D warnings` 通过。

## 实际结果

白头对比是中心亮度与固定环形皮肤的差，表中取绝对值；它描述局部凸起减弱，不自动换算审美分数。

| 样片/档位 | v3 白头亮度对比 | v4c 白头亮度对比 | 观察 |
|---|---:|---:|---|
| oily / natural | 53.10 | 36.73 | 明显减弱，仍能看到凸起 |
| oily / strong | 48.80 | 24.44 | 减弱约一半，连续过渡；局部材质仍不等同精修 |
| oily / texture | 52.74 | 30.51 | 保留周围细节，白头仍有残留 |

- teen 三档与 v3 输出逐字节相同；红丘疹、痘印没有额外改善。
- cystic 三档与 v3 输出逐字节相同；囊肿、深层痘印仍失败。
- 九项选区 ROI 外和 alpha 变化均为 0；选区外标注负例保持。
- oily strong 相对 v3 新增 801 个变化像素；主要落在明显白头，正常大片油脂高光整体仍保留，但少数紧凑闪点可能被误修。第一版新增约 5,013 个像素的更广泛油脂闪点修改已被收紧；这些历史候选不作为当前生产证据。
- native CLI 总耗时 149–877ms，含输入/协议/输出和同时编译的干扰；没有证明浏览器完整热运行效率。

代表图：[自然档](skin-bright-v4c/acne-oily-natural-100pct.png)、[强档](skin-bright-v4c/acne-oily-strong-100pct.png)、[纹理档](skin-bright-v4c/acne-oily-texture-100pct.png)、[6×白头](skin-bright-v4c/whitehead-strong-6x.png)、[相对v3新增区域](skin-bright-v4c/changed-overlay.png)。图中比较原图/旧 baseline/v3/v4c，不能将旧 baseline 的蜡感误归因到 v4c。

## 决定与下一步

不将亮斑统计默认加入磨皮，避免把真实油脂反射当作瑕疵。强磨皮蜡感没有在本实验中通过新的多图质感验收，囊肿几何也没有重建。后续先把用户明确圈选的白头作为局部修复区域，与保留细节的 ring/patch donor 比较，再加入皮肤材质/瑕疵分类；自动全脸处理仍需真实高光负例和更多独立人物验证。
