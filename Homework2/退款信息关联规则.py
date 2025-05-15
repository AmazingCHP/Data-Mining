import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 读取 CSV 文件
file_path = '10G/4关联.csv'
df = pd.read_csv(file_path)

# 2. 转换 frozenset 为字符串
df['antecedents'] = df['antecedents'].str.replace("frozenset\\(|[{}']|\\)", '', regex=True)
df['consequents'] = df['consequents'].str.replace("frozenset\\(|[{}']|\\)", '', regex=True)

# 3. 筛选包含“已退款”或“部分退款”的规则
refund_keywords = ['已退款', '部分退款']
refund_df = df[df['antecedents'].str.contains('|'.join(refund_keywords)) |
               df['consequents'].str.contains('|'.join(refund_keywords))].copy()

# 4. 添加可视化用的规则文本
refund_df['rule'] = refund_df['antecedents'] + " → " + refund_df['consequents']

# 5. 获取置信度前30的规则
top_n = 30
top_rules = refund_df.sort_values(by='confidence', ascending=False).head(top_n)

# 6. 绘图
plt.figure(figsize=(12, 8))
sns.barplot(data=top_rules, y='rule', x='confidence', color='skyblue')

plt.title('前30条包含退款的高置信度关联规则', fontsize=16)
plt.xlabel('置信度 (Confidence)')
plt.ylabel('规则 (前件 → 后件)')
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("包含退款的前30条关联规则.png", dpi=300)
plt.show()
