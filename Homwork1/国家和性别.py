import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
df = pd.read_parquet("10G_data/part-00000.parquet")

# 2. 统计各国家的性别数量
gender_counts = df.groupby(['country', 'gender']).size().unstack()

# 按总用户数排序国家（取前20名避免图表拥挤）
top_countries = gender_counts.sum(axis=1).sort_values(ascending=False).head(20).index
gender_counts = gender_counts.loc[top_countries]

# 3. 绘制并排柱状图
plt.figure(figsize=(15, 8))
bar_width = 0.35  # 柱子宽度
x = np.arange(len(gender_counts.index))  # x轴位置

# 绘制男性柱子
plt.bar(
    x - bar_width / 2,
    gender_counts['男'],
    width=bar_width,
    label='男',
    color='#1f77b4',
    edgecolor='black'
)

# 绘制女性柱子
plt.bar(
    x + bar_width / 2,
    gender_counts['女'],
    width=bar_width,
    label='女',
    color='#ff7f0e',
    edgecolor='black'
)

# 4. 添加图表元素
plt.title('各国用户性别分布（Top 20国家）', fontsize=16, pad=20)
plt.xlabel('国家', fontsize=12)
plt.ylabel('用户数量', fontsize=12)
plt.xticks(x, gender_counts.index, rotation=45, ha='right')  # 旋转45度避免重叠
plt.legend(title='性别', bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在外侧

# 5. 添加柱子数值标签（仅显示超过50的标签）
for i in x:
    if gender_counts['男'][i] > 50:
        plt.text(
            i - bar_width / 2, gender_counts['男'][i] + 5,
            f"{gender_counts['男'][i]:,}",  # 千位分隔符
            ha='center', va='bottom',
            fontsize=9
        )
    if gender_counts['女'][i] > 50:
        plt.text(
            i + bar_width / 2, gender_counts['女'][i] + 5,
            f"{gender_counts['女'][i]:,}",
            ha='center', va='bottom',
            fontsize=9
        )

plt.tight_layout()
plt.show()

# 输出统计表格
print("\n各国性别分布统计（Top 20）：")
print(gender_counts)
