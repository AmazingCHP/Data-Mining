import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 读取 CSV 文件
file_path = '10G/1关联.csv'
df = pd.read_csv(file_path)

# 2. 清洗 frozenset 格式为可读的文本
df['antecedents'] = df['antecedents'].str.replace("frozenset\\(|[{}']|\\)", '', regex=True)
df['consequents'] = df['consequents'].str.replace("frozenset\\(|[{}']|\\)", '', regex=True)

# 3. 选择置信度最高的前 N 条规则
top_n = 30
top_rules = df.sort_values(by='confidence', ascending=False).head(top_n)

# 4. 构造规则文本列
top_rules['规则'] = top_rules['antecedents'] + " → " + top_rules['consequents']

# 5. 绘制柱状图
plt.figure(figsize=(12, 8))
sns.barplot(data=top_rules, y='规则', x='confidence', color='steelblue')

plt.title('Top 20 高置信度商品类别关联规则', fontsize=16)
plt.xlabel('置信度')
plt.ylabel('规则')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("高置信度商品类别关联规则.png", dpi=300)
plt.show()
