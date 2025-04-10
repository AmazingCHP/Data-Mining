import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置中文字体
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 输入输出路径设置
input_directory = '30processed_data'
output_directory = 'pictures/ppppp'

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 获取所有 .parquet 文件
parquet_files = [f for f in os.listdir(input_directory) if f.endswith('4.parquet')]

# 初始化总数据 DataFrame
all_data = pd.DataFrame()

# 加载数据
for file in tqdm(parquet_files, desc="加载数据文件", ncols=100):
    file_path = os.path.join(input_directory, file)
    df = pd.read_parquet(file_path)
    all_data = pd.concat([all_data, df], ignore_index=True)

sampled_df = all_data.sample(frac=0.1, random_state=42)  # 0.1% 抽样

# 生成交叉表
sample_gender_country = pd.crosstab(sampled_df['country'], sampled_df['gender'])

# 计算每行的总和（即每个国家的总人数）
country_totals = sample_gender_country.sum(axis=1)

# 计算百分比
sample_gender_country_percentage = sample_gender_country.divide(country_totals, axis=0) * 100

# Top10国家
sample_top_countries = sample_gender_country_percentage.sum(axis=1).nlargest(10).index
sample_gender_country_top10_percentage = sample_gender_country_percentage.loc[sample_top_countries]

# 绘图保存
plt.figure(figsize=(12, 6))
sample_gender_country_top10_percentage.plot(kind='bar', stacked=True, colormap='Pastel2')
plt.title('各国用户性别分布（前10国家） - 百分比', fontsize=15)
plt.xlabel('国家', fontsize=12)
plt.ylabel('百分比 (%)', fontsize=12)
plt.legend(title='性别')
plt.tight_layout()

output_path2 = os.path.join(output_directory, '30gender_country_distribution_top10_sampled_percentage.png')
plt.savefig(output_path2)
plt.close()

print("图表已保存：")
print(f"数据图（百分比）：{output_path2}")
