import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from tqdm import tqdm  # 导入tqdm库

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 根据你的系统选择合适的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 假设processed_data目录中包含.parquet文件
input_directory = 'processed_data'  # 你要处理的文件夹路径
output_directory = 'pictures/ppppp'  # 设置保存图片的文件夹路径

# 创建output_directory文件夹（如果不存在的话）
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 遍历所有parquet文件，加载数据
parquet_files = [f for f in os.listdir(input_directory) if f.endswith('5.parquet')]

# 初始化空的DataFrame来存储所有数据
all_data = pd.DataFrame()

# 使用tqdm显示进度条加载数据
for file in tqdm(parquet_files, desc="加载文件", ncols=100):
    file_path = os.path.join(input_directory, file)
    df = pd.read_parquet(file_path)

    # 将数据追加到all_data中
    all_data = pd.concat([all_data, df], ignore_index=True)

# 处理数据：从全部数据中抽样（例如，抽取10%的数据）
sampled_df = all_data.sample(frac=0.1, random_state=42)  # 10%的抽样，random_state保证重复性

# 统计每个省份和消费类别的分布
category_counts_by_province = sampled_df.groupby(['province', 'category']).size().unstack(fill_value=0)

# 计算百分比
category_counts_percent = category_counts_by_province.div(category_counts_by_province.sum(axis=1), axis=0) * 100

# 为每个省份生成图表
for province, data in category_counts_percent.iterrows():
    plt.figure(figsize=(12, 8))
    sns.barplot(x=data.index, y=data.values)
    plt.title(f'{province}的消费类别分布')
    plt.xlabel('消费类别')
    plt.ylabel('百分比 (%)')
    plt.xticks(rotation=45)

    # 显示百分比标签
    for i, v in enumerate(data.values):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # 保存每个省份的图像到指定文件夹
    output_file = os.path.join(output_directory, f'{province}_consumption_distribution.png')
    plt.savefig(output_file)
    plt.close()

print("所有省份的消费类别分布图已生成。")
