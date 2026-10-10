# 历史 NACA0012 通过结论已撤回

本目录是旧计算记录，原文件保留。其图上估读参考 Cl=0.205、Cd=0.120 不能继续用于严格3%验收。

本轮从[原论文图10/11](https://arxiv.org/html/2006.10487)的 present-study 矢量曲线提取攻角4°的标记，得到 Cl=0.2009636030、Cd=0.1250065462，各采用 ±0.0001 图形读取区间。该区间只覆盖坐标读取舍入，不覆盖作者计算误差。旧结果 Cd=0.1209791959 对新参考的名义误差约3.22%，不能标为达标。

提取程序：[extract_naca_reference.py](../../scripts/extract_naca_reference.py)；[读取事实与源页面哈希](../naca-reference-extraction.json)；[三网格实际验证](../verification-external-naca/report.md)。没有匹配攻角4°的公开 Cp 表，压力系数曲线不作3%参考验收。
