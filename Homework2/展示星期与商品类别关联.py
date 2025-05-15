import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 设置中文字体和其他配置
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 读取CSV文件
file_path = '10G/星期关联规则.csv'
df = pd.read_csv(file_path)

# 2. 将frozenset转换为字符串
df['antecedents'] = df['antecedents'].str.replace("frozenset\\(|[{}']|\\)", '', regex=True)
df['consequents'] = df['consequents'].str.replace("frozenset\\(|[{}']|\\)", '', regex=True)

# 3. 筛选包含星期信息的关联规则
days_of_week = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
df = df[(df['antecedents'].isin(days_of_week) & df['consequents'].isin(days_of_week)) |
        (~df['antecedents'].isin(days_of_week) & df['consequents'].isin(days_of_week)) |
        (df['antecedents'].isin(days_of_week) & ~df['consequents'].isin(days_of_week))]

# 4. 创建可视化规则文本
df['rule'] = df['antecedents'] + " → " + df['consequents']

# 5. 获取置信度最高的前30条规则
top_n = 30
top_rules = df.sort_values(by='confidence', ascending=False).head(top_n)

# 确保数据量为30条
if len(top_rules) < 30:
    top_n = len(top_rules)
    print(f"实际符合条件的规则数量为 {top_n} 条，将打印全部规则。")
else:
    print(f"将打印前30条高置信度关联规则。")

# 6. 打印前30条规则
print("前30条高置信度关联规则（仅含星期信息）:")
print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(top_n))

# 7. 绘制条形图
plt.figure(figsize=(12, 10))
sns.barplot(data=top_rules, y='rule', x='confidence', color='skyblue')

plt.title('前30条高置信度关联规则（仅含星期信息）', fontsize=16)
plt.xlabel('置信度 (Confidence)')
plt.ylabel('关联规则 (前件 → 后件)')
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("互评作业2图片/星期与商品类别关联.png", dpi=300)
plt.show()