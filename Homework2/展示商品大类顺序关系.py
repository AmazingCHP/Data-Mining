import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
data = pd.read_csv('10G/商品大类顺序关系.csv')

# 按支持度排序并取前30条数据
top_30 = data.nlargest(30, 'support')

# 准备数据
categories = top_30['from'] + ' -> ' + top_30['to']  # 合并起始和目标类别
supports = top_30['support']

# 创建横向条形图
plt.figure(figsize=(10, 12))
plt.barh(categories, supports, color='skyblue')

# 添加标题和标签
plt.xlabel('支持度 (Support)', fontsize=12)
plt.ylabel('商品大类顺序关系', fontsize=12)
plt.title('支持度前30的商品大类顺序关系', fontsize=14)

# 调整布局
plt.tight_layout()

plt.savefig("展示商品大类顺序关系.png", dpi=300)
# 显示图形
plt.show()
