import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 输入输出路径
input_directory = '30processed_data111'
output_directory = 'pictures/ppppp30'

# 创建输出文件夹
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 获取.parquet文件列表
all_parquet_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]

# 随机抽取文件
num_samples = min(100, len(all_parquet_files))
parquet_files = random.sample(all_parquet_files, num_samples)

# 初始化总数据
all_data = pd.DataFrame()

# 加载数据
for file in tqdm(parquet_files, desc="加载文件", ncols=100):
    file_path = os.path.join(input_directory, file)
    df = pd.read_parquet(file_path)
    all_data = pd.concat([all_data, df], ignore_index=True)

# 抽样数据
sampled_df = all_data.sample(frac=1, random_state=42)

# # 收入放大10倍（可选）
# sampled_df['income'] = sampled_df['income'] * 10

# 提取需要聚类的特征
data_for_clustering = sampled_df[['income', 'average_price']].dropna()

# 归一化
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data_for_clustering)

# KMeans聚类
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(normalized_data)
sampled_df.loc[data_for_clustering.index, 'cluster'] = clusters

# 绘图
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=normalized_data[:, 0],
    y=normalized_data[:, 1],
    hue=clusters,
    palette='Set1',
    s=100,
    marker='o'
)

# 聚类中心
centers = kmeans.cluster_centers_
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c='black',
    s=200,
    marker='X',
    label='聚类中心'
)

# 图表设置
plt.title('收入与消费金额关系散点图')
plt.xlabel('收入（归一化）')
plt.ylabel('消费金额（归一化）')
plt.legend()
plt.tight_layout()

# 保存图像
output_file = os.path.join(output_directory, 'normalized_income_vs_average_price_with_clusters.png')
plt.savefig(output_file)
plt.close()

print("归一化后的收入与消费金额散点图已生成，保存在 'pictures/ppppp30' 文件夹下。")
