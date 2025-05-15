import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 读取 CSV 文件
file_path = '10G/1关联.csv'  # 根据实际路径修改
df = pd.read_csv(file_path)

# 2. 转换 frozenset 为可读文本
df['antecedents'] = df['antecedents'].str.replace("frozenset\\(|[{}']|\\)", '', regex=True)
df['consequents'] = df['consequents'].str.replace("frozenset\\(|[{}']|\\)", '', regex=True)

# 3. 筛选前件或后件中包含“电子产品”的规则
df_filtered = df[(df['antecedents'].str.contains('电子产品')) | (df['consequents'].str.contains('电子产品'))]

# 4. 按置信度排序，取前 N 条规则
top_n = 30
top_rules = df_filtered.sort_values(by='confidence', ascending=False).head(top_n)

# 5. 构造规则显示列
top_rules['rule'] = top_rules['antecedents'] + " → " + top_rules['consequents']

# 6. 可视化
plt.figure(figsize=(12, 8))
sns.barplot(data=top_rules, y='rule', x='confidence', palette='YlOrBr')

plt.title('包含“电子产品”的高置信度关联规则', fontsize=16)
plt.xlabel('置信度 (Confidence)')
plt.ylabel('规则 (前件 → 后件)')
plt.tight_layout()
plt.grid(True, axis='x', linestyle='--', alpha=0.7)

plt.savefig("包含电子产品的高置信度关联规则.png", dpi=300)
# 7. 显示图像并保存
plt.show()
