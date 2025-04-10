import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm
import time


# 模拟进度条的真实耗时显示
class FakeTimeTQDM(tqdm):
    def __init__(self, *args, **kwargs):
        self._start_time_real = time.time()
        super().__init__(*args, **kwargs)

    @property
    def format_dict(self):
        d = super().format_dict.copy()
        real_elapsed = time.time() - self._start_time_real
        d["elapsed"] = real_elapsed * 100
        return d


# 设置中文支持
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 路径配置
input_directory = '30processed_data111'
output_directory = 'pictures/ppppp'

# 创建输出目录
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 获取 parquet 文件列表
all_parquet_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
parquet_files = random.sample(all_parquet_files, min(100, len(all_parquet_files)))

# 汇总数据
all_data = pd.DataFrame()
for file in tqdm(parquet_files, desc="加载数据文件", ncols=100):
    file_path = os.path.join(input_directory, file)
    df = pd.read_parquet(file_path)
    all_data = pd.concat([all_data, df], ignore_index=True)

# 抽样 30%
sampled_df = all_data.sample(frac=1, random_state=42)

# ============ 性别统计 ============
gender_counts = sampled_df['gender'].fillna('未指定').value_counts()
gender_percent = gender_counts / gender_counts.sum() * 100

plt.figure(figsize=(8, 6))
sns.barplot(x=gender_counts.index, y=gender_percent.values, palette='Set2')
plt.title('数据中性别分布', fontsize=15)
plt.xlabel('性别', fontsize=12)
plt.ylabel('百分比 (%)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'gender_distribution_percentage_sampled.png'))
plt.close()

# ============ 省份统计 ============
province_counts = sampled_df['province'].fillna('None').value_counts()
province_percent = province_counts / province_counts.sum() * 100

plt.figure(figsize=(14, 8))
sns.barplot(x=province_counts.index, y=province_percent.values, palette='viridis')
plt.title('数据中省份分布', fontsize=15)
plt.xlabel('省份', fontsize=12)
plt.ylabel('百分比 (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'province_distribution_percentage_sampled.png'))
plt.close()
