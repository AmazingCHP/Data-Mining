import pandas as pd
import pyarrow.parquet as pq
import os
import json
import gc
import psutil
from collections import defaultdict


def mem_usage():
    """返回当前进程内存使用量(MB)"""
    return psutil.Process().memory_info().rss / (1024 ** 2)


def get_all_categories(input_dir):
    """预扫描所有文件获取完整的分类变量取值范围"""
    category_values = defaultdict(set)
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
             if f.startswith('part-') and f.endswith('.parquet')]

    # 只扫描前几个文件获取大致分类范围
    for file in files[:2]:  # 调整这个数字以平衡准确性和性能
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(batch_size=10000):
            df = batch.to_pandas()

            # 收集分类变量的所有可能值
            if 'category' in df.columns:
                category_values['category'].update(df['category'].unique())
            if 'gender' in df.columns:
                category_values['gender'].update(df['gender'].unique())
            if 'country' in df.columns:
                category_values['country'].update(df['country'].unique())
            if 'province' in df.columns:
                category_values['province'].update(df['province'].str.extract(r'^(.*?省)')[0].unique())

    # 转换为排序后的列表，确保顺序一致
    return {k: sorted(v) for k, v in category_values.items()}


def process_chunk(df, all_categories):
    """处理单个数据块的函数"""
    # 转换时间类型并统一时区
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df['registration_date'] = pd.to_datetime(df['registration_date']).dt.tz_localize(None)

    # 处理JSON字段 - 使用更安全的方式
    try:
        df['purchase_history'] = df['purchase_history'].apply(json.loads)
        purchase_df = pd.json_normalize(df['purchase_history'])
        df = pd.concat([df.drop('purchase_history', axis=1), purchase_df], axis=1)
        df['items_count'] = df['items'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df = df.drop('items', axis=1)
    except Exception as e:
        print(f"JSON处理错误: {e}")
        df['items_count'] = 0

    # 处理缺失值
    median_age = df['age'].median()
    median_income = df['income'].median()
    median_credit = df['credit_score'].median()

    df = df.assign(
        age=df['age'].fillna(median_age),
        income=df['income'].fillna(median_income),
        credit_score=df['credit_score'].fillna(median_credit),
        gender=df['gender'].fillna('未知')
    )

    # 处理异常值
    df = df[(df['age'] > 0) & (df['age'] < 120)]

    Q1 = df['income'].quantile(0.25)
    Q3 = df['income'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['income'] < (Q1 - 1.5 * IQR)) | (df['income'] > (Q3 + 1.5 * IQR)))]  # 去除收入异常值

    df = df[(df['credit_score'] >= 300) & (df['credit_score'] <= 850)]  # 去除信用分异常值

    # 特征工程
    df['province'] = df['chinese_address'].str.extract(r'^(.*?省)')
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # 计算注册时间到当前时间的天数
    df['days_since_registration'] = (df['timestamp'] - df['registration_date']).dt.days

    # 返回经过处理的数据
    return df


def process_file(input_file, output_dir, all_categories, chunk_size=50000):
    """处理单个文件"""
    print(f"正在处理文件: {input_file}")

    # 创建Parquet文件读取器
    parquet_file = pq.ParquetFile(input_file)

    # 分批读取和处理
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
        print(f"处理批次 {i + 1}, 内存使用: {mem_usage():.2f} MB")

        try:
            # 转换为DataFrame
            df = batch.to_pandas()

            # 处理数据块
            processed_df = process_chunk(df, all_categories)

            # 保存结果
            output_file = os.path.join(output_dir,
                                       f"{os.path.splitext(os.path.basename(input_file))[0]}_part{i}.parquet")
            processed_df.to_parquet(output_file)

        except Exception as e:
            print(f"处理批次 {i + 1} 时出错: {e}")
            continue

        finally:
            # 确保释放内存
            del df, processed_df
            gc.collect()


def main():
    input_dir = '30G_data/'
    output_dir = '30processed_data111/'
    os.makedirs(output_dir, exist_ok=True)

    # 预扫描获取所有分类变量的可能取值
    print("正在预扫描数据以获取分类变量范围...")
    all_categories = get_all_categories(input_dir)
    print("发现以下分类变量范围:")
    for k, v in all_categories.items():
        print(f"{k}: {v}")

    # 获取所有输入文件
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
             if f.startswith('part-') and f.endswith('.parquet')]

    # 处理每个文件
    for file in files:
        process_file(file, output_dir, all_categories)

    print("所有文件处理完成！")


if __name__ == "__main__":
    main()
