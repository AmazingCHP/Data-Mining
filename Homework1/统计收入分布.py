import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sphinx.util.console import black
from tqdm import tqdm
from tqdm import tqdm
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


# 设置中文字体
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 输入输出路径设置
input_directory = 'processed_data'
output_directory = 'pictures/ppppp'

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 获取所有 .parquet 文件
parquet_files = [f for f in os.listdir(input_directory) if f.endswith('4.parquet')]

# 初始化总数据 DataFrame
all_data = pd.DataFrame()

# 加载数据
for file in FakeTimeTQDM(parquet_files, desc="加载数据文件", ncols=100):
    file_path = os.path.join(input_directory, file)
    df = pd.read_parquet(file_path)
    all_data = pd.concat([all_data, df], ignore_index=True)

# 抽样
sampled_df = all_data.sample(frac=0.1, random_state=42)  # 10% 抽样

# 统计收入的分布并计算百分比
income_distribution = sampled_df['income'].value_counts(normalize=True).sort_index() * 100  # 计算百分比并排序

# 绘制收入分布的折线图
plt.figure(figsize=(12, 6))
income_distribution.plot(kind='line', marker='o', color='skyblue', markersize=4)  # 调整数据点的大小
plt.title('数据中收入分布折线图', fontsize=15)
plt.xlabel('收入', fontsize=12)
plt.ylabel('百分比 (%)', fontsize=12)
plt.ylim(0, income_distribution.max() * 1.1)  # 确保纵轴从0开始，并稍微扩大上限
plt.tight_layout()

# 保存图表
output_path2 = os.path.join(output_directory, 'income_distribution_sampled_percentage.png')
plt.savefig(output_path2)
plt.close()

print("图表已保存：")
print(f"数据图（收入分布百分比折线图）：{output_path2}")
