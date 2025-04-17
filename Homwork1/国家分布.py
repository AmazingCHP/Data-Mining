import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 支持中文显示

def get_all_files(folder_path, suffix=".parquet"):
    """获取指定文件夹下所有指定后缀的文件路径"""
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(suffix):
                file_paths.append(os.path.join(root, file))
    return file_paths

def analyze_country_distribution(folder_path):
    """分析国家分布"""
    files = get_all_files(folder_path)
    all_countries = []

    for file in tqdm(files, desc="分析国家分布中"):
        try:
            df = pd.read_parquet(file, columns=['country'], engine='pyarrow')
            all_countries.extend(df['country'].dropna().astype(str).tolist())
        except Exception as e:
            print(f"读取文件出错 {file}: {e}")
            continue

    if all_countries:
        country_series = pd.Series(all_countries)
        country_counts = country_series.value_counts()

        # 只显示前20个国家（如国家种类太多）
        top_countries = country_counts.head(20)

        # 绘制柱状图
        plt.figure(figsize=(12, 6))
        ax = top_countries.plot(
            kind='bar',
            color='lightgreen',
            edgecolor='black'
        )
        ax.set_title("国家分布（Top 20）", fontsize=14)
        ax.set_xlabel("国家", fontsize=12)
        ax.set_ylabel("人数", fontsize=12)
        plt.xticks(rotation=45, ha='right')

        # 添加数值标签
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom',
                fontsize=10, xytext=(0, 5),
                textcoords='offset points'
            )

        plt.tight_layout()
        plt.show()
    else:
        print("未提取到任何国家数据")

# 执行分析
analyze_country_distribution('10G_data_new')
