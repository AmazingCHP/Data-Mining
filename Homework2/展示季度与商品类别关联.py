import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Set Chinese font and other configurations
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. Read CSV file
file_path = '10G/3季度关联.csv'
df = pd.read_csv(file_path)

# 2. Convert frozenset to string
df['antecedents'] = df['antecedents'].str.replace("frozenset\\(|[{}']|\\)", '', regex=True)
df['consequents'] = df['consequents'].str.replace("frozenset\\(|[{}']|\\)", '', regex=True)

# 3. Create rule text for visualization
df['rule'] = df['antecedents'] + " → " + df['consequents']

# 4. Get top 30 rules by confidence
top_n = 30
top_rules = df.sort_values(by='confidence', ascending=False).head(top_n)

# 5. Plot
plt.figure(figsize=(12, 10))
sns.barplot(data=top_rules, y='rule', x='confidence', color='skyblue')

plt.title('前30条高置信度关联规则', fontsize=16)
plt.xlabel('置信度 (Confidence)')
plt.ylabel('关联规则 (前件 → 后件)')
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("季度与商品类别关联.png", dpi=300)
plt.show()
