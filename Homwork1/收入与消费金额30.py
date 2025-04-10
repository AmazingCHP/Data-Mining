import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
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
num_samples = min(100, len(all_parquet_files))
parquet_files = random.sample(all_parquet_files, num_samples)

# 初始化一个空的DataFrame，准备存储数据
combined_data = pd.DataFrame()

# 加载数据，逐个文件处理，减少内存压力
for file in tqdm(parquet_files, desc="加载文件", ncols=100):
    file_path = os.path.join(input_directory, file)

    # 加载数据时只选择 'income', 'age', 'category' 三列
    try:
        df = pd.read_parquet(file_path, columns=['income', 'age', 'category'])
        combined_data = pd.concat([combined_data, df], ignore_index=True)
    except Exception as e:
        print(f"跳过文件 {file}，错误：{e}")
        continue

# 去除空值
combined_data = combined_data.dropna(subset=['income', 'age', 'category'])

# 绘制散点图：收入 vs 年龄，颜色代表消费类别
plt.figure(figsize=(10, 8))
sns.scatterplot(data=combined_data, x='age', y='income', hue='category', palette='Set2', alpha=0.7, edgecolor='w',
                linewidth=0.5)

plt.title('收入与年龄的关系（按消费类别着色）')
plt.xlabel('年龄')
plt.ylabel('收入')
plt.legend(title='消费类别', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# 保存图像
output_path = os.path.join(output_directory, 'income_vs_age_by_category.png')
plt.savefig(output_path)
plt.close()

print(f"收入与年龄的散点图已生成，保存在：{output_path}")
