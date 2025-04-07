import os
import pandas as pd
import pyarrow.parquet as pq
import random
import matplotlib.pyplot as plt

# 设置输入输出路径
input_dir = 'processed_data/'
output_dir = 'pictures/pprovince/'

# 创建输出文件夹（如果不存在）
os.makedirs(output_dir, exist_ok=True)


def load_and_clean_data(input_dir):
    """加载并清洗数据"""
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.parquet')]
    data_frames = []

    for file in files:
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(batch_size=50000):
            df = batch.to_pandas()

            # 去除不符合要求的性别数据
            df = df[df['gender'].isin(['男', '女'])]

            # 只保留需要的字段
            df = df[['province', 'income', 'average_price']]

            data_frames.append(df)

    # 合并所有数据
    return pd.concat(data_frames, ignore_index=True)


def sample_data(df, sample_size=0.1):
    """从数据中抽取随机样本"""
    return df.sample(frac=sample_size, random_state=42)


def plot_income_difference(df, output_file):
    """按省份绘制年收入差异图"""
    income_by_province = df.groupby('province')['income'].median().sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    income_by_province.plot(kind='bar', color='skyblue')
    plt.title('Province-wise Annual Income Difference')
    plt.xlabel('Province')
    plt.ylabel('Annual Income (Median)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_average_price_difference(df, output_file):
    """按省份绘制平均消费金额差异图"""
    avg_price_by_province = df.groupby('province')['average_price'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    avg_price_by_province.plot(kind='bar', color='lightgreen')
    plt.title('Province-wise Average Price Difference')
    plt.xlabel('Province')
    plt.ylabel('Average Price')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def main():
    # 加载并清洗数据
    df = load_and_clean_data(input_dir)

    # 从数据中抽样
    sampled_df = sample_data(df)

    # 绘制并保存图表
    income_output_file = os.path.join(output_dir, 'income_difference.png')
    plot_income_difference(sampled_df, income_output_file)

    avg_price_output_file = os.path.join(output_dir, 'average_price_difference.png')
    plot_average_price_difference(sampled_df, avg_price_output_file)

    print("图片已保存至:", output_dir)


if __name__ == "__main__":
    main()
