import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 中文字体支持
pd.set_option('display.max_columns', None)


def get_all_files(folder_path, suffix=".parquet"):
    """获取所有指定后缀的文件路径"""
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(suffix):
                file_paths.append(os.path.join(root, file))
    return file_paths


def analyze_age_distribution(folder_path):
    """分析年龄分布，仅提取 age 列，节省内存"""
    files = get_all_files(folder_path)
    all_ages = []

    for file in tqdm(files, desc="分析年龄分布中"):
        try:
            df = pd.read_parquet(file, columns=['age'], engine='pyarrow')
            all_ages.extend(df['age'].dropna().astype(int).tolist())
        except Exception as e:
            print(f"读取文件出错 {file}: {e}")
            continue

    if all_ages:
        age_series = pd.Series(all_ages)
        plt.figure(figsize=(10, 6))
        ax = age_series.value_counts().sort_index().plot(
            kind='bar',
            color='lightgreen',
            edgecolor='black'
        )
        ax.set_title("年龄分布统计")
        ax.set_xlabel("年龄")
        ax.set_ylabel("人数")
        plt.xticks(rotation=45, ha='right')

        # 添加数值标签
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom',
                fontsize=8, xytext=(0, 3),
                textcoords='offset points'
            )
        plt.tight_layout()
        plt.show()
    else:
        print("未提取到任何年龄数据")


# 调用分析函数
analyze_age_distribution('10G_data_new')
