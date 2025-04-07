import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # 导入tqdm库
from sklearn.cluster import KMeans  # 导入KMeans聚类算法
import random

# 设置中文字体
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 根据你的系统选择合适的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 假设processed_data目录中包含.parquet文件
input_directory = '30processed_data'  # 你要处理的文件夹路径
output_directory = 'pictures/ppppp30'  # 设置保存图片的文件夹路径

# 创建output_directory文件夹（如果不存在的话）
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 遍历所有parquet文件，加载数据
all_parquet_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]

# 设置抽样数量，比如随机抽取10个文件，注意避免超过实际数量
num_samples = min(40, len(all_parquet_files))
parquet_files = random.sample(all_parquet_files, num_samples)

# 初始化空的DataFrame来存储所有数据
all_data = pd.DataFrame()

# 使用tqdm显示进度条加载数据
for file in tqdm(parquet_files, desc="加载文件", ncols=100):
    file_path = os.path.join(input_directory, file)
    df = pd.read_parquet(file_path)

    # 将数据追加到all_data中
    all_data = pd.concat([all_data, df], ignore_index=True)

# 处理数据：从全部数据中抽样（例如，抽取0.01%的数据）
sampled_df = all_data.sample(frac=0.01, random_state=42)  # 0.01%的抽样，random_state保证重复性

# 将收入字段的值扩大10倍（如果需要的话）
sampled_df['income'] = sampled_df['income'] * 10  # 假设收入需要扩大10倍

# 提取需要进行聚类的特征（收入和消费金额）
data_for_clustering = sampled_df[['income', 'average_price']].dropna()

# 使用KMeans进行聚类，设定聚类的数量（比如7个聚类）
kmeans = KMeans(n_clusters=7, random_state=42)
sampled_df['cluster'] = kmeans.fit_predict(data_for_clustering)

# 使用Seaborn绘制收入与消费金额的散点图，聚类结果不同的类用不同的颜色标记
plt.figure(figsize=(10, 6))
sns.scatterplot(x='income', y='average_price', hue='cluster', data=sampled_df, palette='Set1', s=100, marker='o')

# 绘制聚类中心
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X', label='聚类中心')

# 设置标题和标签
plt.title('收入与消费金额之间的关系及聚类结果')
plt.xlabel('收入 (已放大10倍)')
plt.ylabel('消费金额（average_price）')

# 显示图表
plt.legend()
plt.tight_layout()

# 保存聚类后的散点图到指定文件夹
output_file = os.path.join(output_directory, 'income_vs_average_price_with_clusters_scaled.png')
plt.savefig(output_file)
plt.close()

print("收入与消费金额的散点图已生成，保存在 'pictures/ppppp' 文件夹下。")
