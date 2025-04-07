import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np
import gc
from tqdm import tqdm
import os
from glob import glob
import random

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义加载数据函数
def load_data(file_pattern, sample_size=10):
    files = glob(file_pattern)
    if len(files) < sample_size:
        sample_files = files
    else:
        sample_files = random.sample(files, sample_size)  # 随机抽取部分文件

    for file in tqdm(sample_files, desc="读取进度"):
        df = pd.read_parquet(file)
        yield df
        del df
        gc.collect()


def credit_score_vs_age(df):
    """信用评分与用户年龄的关系分析"""
    # 假设数据中有 'age' 列，表示用户的年龄
    bins = np.arange(0, df['age'].max() + 10, 10)
    labels = [f'{i}-{i + 9}' for i in bins[:-1]]

    df['age_range'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    # 可视化：年龄段与信用评分的关系
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='age', y='credit_score', data=df, hue='age_range', palette='coolwarm', alpha=0.7)
    plt.title('信用评分与用户年龄的关系', fontsize=15)
    plt.xlabel('用户年龄', fontsize=12)
    plt.ylabel('信用评分', fontsize=12)
    plt.tight_layout()
    plt.savefig('pictures/ppppp30/credit_score_vs_age.png')  # 保存图片
    plt.close()

    return df[['age', 'credit_score', 'age_range']]


# 执行数据分析，包含年龄与信用评分的关系
def analyze_data_with_age(file_pattern='30processed_data/*.parquet'):
    """执行数据分析（包括年龄与信用评分的关系）"""
    credit_activity_data = []
    registration_consumption_data = []
    age_credit_score_data = []

    for df in load_data(file_pattern, sample_size=100):
        # 6. 信用评分与年龄的关系
        age_credit_score_data.append(credit_score_vs_age(df))

        del df
        gc.collect()

    # 合并结果
    credit_activity_combined = pd.concat(credit_activity_data, ignore_index=True)
    registration_avg_price_combined = pd.concat([x[0] for x in registration_consumption_data], ignore_index=True)
    registration_category_combined = pd.concat([x[1] for x in registration_consumption_data], ignore_index=True)
    age_credit_score_combined = pd.concat(age_credit_score_data, ignore_index=True)

    return {
        'credit_activity': credit_activity_combined,
        'registration_avg_price': registration_avg_price_combined,
        'registration_category': registration_category_combined,
        'age_credit_score': age_credit_score_combined
    }


# 执行分析
results = analyze_data_with_age()

# 打印统计结果
print("=== 信用评分与活跃度的关系 ===")
print(results['credit_activity'].head())
print("\n=== 注册时间与消费金额的关系 ===")
print(results['registration_avg_price'])
print("\n=== 注册时间与消费类别的关系 ===")
print(results['registration_category'])
print("\n=== 信用评分与用户年龄的关系 ===")
print(results['age_credit_score'].head())
