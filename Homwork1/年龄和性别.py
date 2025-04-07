import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
df = pd.read_parquet("10G_data/part-00000.parquet")

# 2. 创建年龄段分组
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
age_labels = []
for i in range(len(age_bins) - 1):
    age_labels.append(str(age_bins[i]) + "-" + str(age_bins[i + 1]))
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

# 3. 统计各年龄段的性别数量
gender_counts = df.groupby(['age_group', 'gender']).size().unstack()

# 4. 绘制并排柱状图
plt.figure(figsize=(12, 6))
bar_width = 0.35  # 柱子宽度
x = np.arange(len(age_labels))  # x轴位置

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

# 5. 添加图表元素
plt.title('各年龄段性别分布（绝对数量）', fontsize=15, pad=20)
plt.xlabel('年龄段', fontsize=12)
plt.ylabel('用户数量', fontsize=12)
plt.xticks(x, age_labels)  # 设置x轴刻度标签
plt.legend(title='性别')

# 6. 添加柱子数值标签
for i in x:
    plt.text(
        i - bar_width / 2, gender_counts['男'][i] + 5,
        str(gender_counts['男'][i]),
        ha='center', va='bottom'
    )
    plt.text(
        i + bar_width / 2, gender_counts['女'][i] + 5,
        str(gender_counts['女'][i]),
        ha='center', va='bottom'
    )

plt.tight_layout()
plt.show()
