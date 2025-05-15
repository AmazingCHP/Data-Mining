import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 文件夹路径设置
input_directory = '30processed_data111'
output_directory = 'pictures/ppppp30'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 随机抽样部分parquet文件
all_parquet_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
num_samples = min(10000, len(all_parquet_files))
parquet_files = random.sample(all_parquet_files, num_samples)

# 初始化一个空的DataFrame，准备存储数据
first_occurrence_data = pd.DataFrame()

# 加载数据，逐个文件处理，减少内存压力
for file in tqdm(parquet_files, desc="加载文件", ncols=100):
    file_path = os.path.join(input_directory, file)

    # 加载数据时只选择 'id' 和 'average_price' 列，减少内存使用
    df = pd.read_parquet(file_path, columns=['id', 'average_price'])

    # 保留每个id的第一次记录
    first_occurrence_data = pd.concat([first_occurrence_data, df]).drop_duplicates(subset='id', keep='first')

    # 处理完一个文件后清理内存，避免占用过多
    del df

# 去除空值
first_occurrence_data = first_occurrence_data.dropna(subset=['average_price'])

# 计算每个用户的总消费金额 (total spending)
total_spending = first_occurrence_data.groupby('id')['average_price'].sum().reset_index()

# 归一化消费金额
scaler = MinMaxScaler()
total_spending['average_price_normalized'] = scaler.fit_transform(total_spending[['average_price']])

# 使用 pd.cut 划分五个区间
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ['0~0.2', '0.2~0.4', '0.4~0.6', '0.6~0.8', '0.8~1.0']
total_spending['average_price_bin'] = pd.cut(total_spending['average_price_normalized'], bins=bins, labels=labels,
                                              include_lowest=True)

# 统计每一档人数
bin_counts = total_spending['average_price_bin'].value_counts().sort_index()

# 绘制柱状图
plt.figure(figsize=(8, 6))
sns.barplot(x=bin_counts.index, y=bin_counts.values, palette='Blues_d')
plt.title('归一化后消费金额区间分布')
plt.xlabel('归一化消费金额区间')
plt.ylabel('人数')
plt.tight_layout()

# 保存图像
output_path = os.path.join(output_directory, 'spending_distribution_binned_first_occurrence.png')
plt.savefig(output_path)
plt.close()

print(f"消费金额区间分布柱状图已生成，保存在：{output_path}")
