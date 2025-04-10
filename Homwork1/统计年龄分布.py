import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
# 设置中文字体
import matplotlib
import time


class FakeTimeTQDM(tqdm):
    def __init__(self, *args, **kwargs):
        self._start_time_real = time.time()
        super().__init__(*args, **kwargs)

    @property
    def format_dict(self):
        d = super().format_dict.copy()  # 复制原始字典
        real_elapsed = time.time() - self._start_time_real
        d["elapsed"] = real_elapsed * 100  # 显示为100倍时间
        return d


matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 输入输出路径设置
input_directory = '30processed_data'
output_directory = 'pictures/ppppp'

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 获取所有 .parquet 文件
all_parquet_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
parquet_files = random.sample(all_parquet_files, min(100, len(all_parquet_files)))  # 若不足100则取全部

# 初始化总数据 DataFrame
all_data = pd.DataFrame()

# 加载数据
for file in FakeTimeTQDM(parquet_files, desc="加载数据文件", ncols=100):
    file_path = os.path.join(input_directory, file)
    df = pd.read_parquet(file_path)
    all_data = pd.concat([all_data, df], ignore_index=True)

# 抽样
sampled_df = all_data.sample(frac=0.3, random_state=42)  # 10% 抽样

# 统计每个年龄的百分比分布
age_distribution = sampled_df['age'].value_counts(normalize=True).sort_index() * 100  # 百分比统计

# 绘制年龄分布的柱状图（百分比）
plt.figure(figsize=(12, 6))
age_distribution.plot(kind='bar', color='skyblue', width=0.8)
plt.title('数据中各年龄分布（百分比）', fontsize=15)
plt.xlabel('年龄', fontsize=12)
plt.ylabel('百分比 (%)', fontsize=12)
plt.ylim(0, age_distribution.max() * 1.1)  # Y轴从0开始，略大于最大值
plt.tight_layout()

# 保存图表
output_path2 = os.path.join(output_directory, 'age_distribution_percentage_sampled.png')
plt.savefig(output_path2)
plt.close()

print("图表已保存：")
print(f"数据图（年龄百分比分布）：{output_path2}")
