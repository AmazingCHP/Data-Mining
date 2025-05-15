import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import gc
from tqdm import tqdm
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 确保保存图片的文件夹存在
if not os.path.exists('pictures'):
    os.makedirs('pictures')


def load_data(file_pattern, chunk_size=10):
    files = glob(file_pattern)
    for i in tqdm(range(0, len(files), chunk_size), desc="读取进度"):
        for file in files[i:i + chunk_size]:
            df = pd.read_parquet(file)
            yield df
            del df
            gc.collect()


def analyze_data(file_pattern='processed_data/*.parquet'):
    """执行数据分析"""
    # 初始化结果容器
    age_counts = pd.Series(dtype='int64')
    gender_counts = pd.Series(dtype='int64')
    country_counts = pd.Series(dtype='int64')
    income_stats = []
    category_stats = pd.Series(dtype='int64')
    province_counts = pd.Series(dtype='int64')
    age_income_data = []

    # 基于实际数据调整的分段设置
    age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    age_labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91+']

    # 修改为每10万元一个分点
    income_bins = [0, 100000, 200000, 300000, 400000, 500000, 600000,
                   700000, 800000, 900000, 1000000, float('inf')]
    income_labels = ['<10万', '10-20万', '20-30万', '30-40万', '40-50万',
                     '50-60万', '60-70万', '70-80万', '80-90万', '90-100万', '>100万']

    # 处理数据（保持不变）
    for df in load_data(file_pattern, chunk_size=10):
        # 1. 年龄分析
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
        age_counts = age_counts.add(df['age_group'].value_counts(), fill_value=0)

        # 2. 性别分析
        gender_counts = gender_counts.add(df['gender'].value_counts(), fill_value=0)

        # 3. 国家分析
        country_counts = country_counts.add(df['country'].value_counts(), fill_value=0)

        # 4. 收入分析（使用新的分段设置）
        df['income_group'] = pd.cut(df['income'], bins=income_bins, labels=income_labels)
        income_stats.append(df.groupby('income_group', observed=False)['income'].describe())

        # 5. 消费类别分析
        category_stats = category_stats.add(df['category'].value_counts(), fill_value=0)

        # 6. 省份分析
        province_counts = province_counts.add(df['province'].value_counts(), fill_value=0)

        # 7. 年龄与收入关系分析
        df['income_group'] = pd.cut(df['income'], bins=income_bins, labels=income_labels)
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
        age_income_data.append(pd.crosstab(df['age_group'], df['income_group'], normalize='index'))

        del df
        gc.collect()

    # 其余代码保持不变...

    # 可视化分析结果 --------------------------------------------------

    # 1. 年龄分布
    plt.figure(figsize=(12, 6))
    age_counts.sort_index().plot(kind='bar', color='#1f77b4')
    plt.title('用户年龄分布', fontsize=15)
    plt.xlabel('年龄区间', fontsize=12)
    plt.ylabel('用户数量', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    for i, v in enumerate(age_counts):
        plt.text(i, v, f"{v:,}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('pictures/age_distribution.png')  # 保存图片
    plt.close()

    # 2. 性别分布
    plt.figure(figsize=(8, 8))
    gender_counts.plot(kind='pie', autopct='%1.1f%%',
                       colors=['#ff7f0e', '#2ca02c', '#d62728'],
                       startangle=90, explode=[0.05] * len(gender_counts))
    plt.title('用户性别比例', fontsize=15)
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('pictures/gender_distribution.png')  # 保存图片
    plt.close()

    # 3. 国家分布
    plt.figure(figsize=(12, 6))
    country_counts.plot(kind='bar', color='#9467bd')
    plt.title('用户国家分布', fontsize=15)
    plt.xlabel('国家', fontsize=12)
    plt.ylabel('用户数量', fontsize=12)
    for i, v in enumerate(country_counts):
        plt.text(i, v, f"{v:,}", ha='center', va='bottom')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('pictures/country_distribution.png')  # 保存图片
    plt.close()

    # 4. 收入分布
    if income_stats:
        income_combined = pd.concat(income_stats).groupby(level=0, observed=False).mean()
        plt.figure(figsize=(12, 6))
        income_combined['count'].plot(kind='bar', color='#8c564b')
        plt.title('用户收入分布', fontsize=15)
        plt.xlabel('收入区间', fontsize=12)
        plt.ylabel('用户数量', fontsize=12)
        for i, v in enumerate(income_combined['count']):
            plt.text(i, v, f"{int(v):,}", ha='center', va='bottom')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('pictures/income_distribution.png')  # 保存图片
        plt.close()

    # 5. 消费类别分布
    if not category_stats.empty:
        plt.figure(figsize=(10, 6))
        category_stats.plot(kind='barh', color='#e377c2')
        plt.title('用户消费类别分布', fontsize=15)
        plt.xlabel('用户数量', fontsize=12)
        plt.ylabel('消费类别', fontsize=12)
        for i, v in enumerate(category_stats):
            plt.text(v, i, f"{v:,}", va='center')
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('pictures/category_distribution.png')  # 保存图片
        plt.close()

    # 6. 省份分布
    if not province_counts.empty:
        plt.figure(figsize=(12, 6))
        province_counts.plot(kind='bar', color='#17becf')
        plt.title('中国用户省份分布', fontsize=15)
        plt.xlabel('省份', fontsize=12)
        plt.ylabel('用户数量', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('pictures/province_distribution.png')  # 保存图片
        plt.close()

    # 7. 年龄与收入关系热力图
    if age_income_data:
        age_income_combined = sum(age_income_data) / len(age_income_data)
        plt.figure(figsize=(12, 8))
        sns.heatmap(age_income_combined, cmap='YlOrRd', annot=True, fmt='.1%', cbar_kws={'label': '占比'})
        plt.title('不同年龄段用户的收入分布', fontsize=15)
        plt.xlabel('收入区间', fontsize=12)
        plt.ylabel('年龄区间', fontsize=12)
        plt.tight_layout()
        plt.savefig('pictures/age_income_heatmap.png')  # 保存图片
        plt.close()

    return {
        'age_distribution': age_counts,
        'gender_distribution': gender_counts,
        'country_distribution': country_counts,
        'income_distribution': income_combined if income_stats else None,
        'category_distribution': category_stats,
        'province_distribution': province_counts
    }


# 执行分析
results = analyze_data()

# 打印统计结果
print("=== 年龄分布统计 ===")
print(results['age_distribution'])
print("\n=== 性别分布统计 ===")
print(results['gender_distribution'])
print("\n=== Top 5国家分布 ===")
print(results['country_distribution'].nlargest(5))
print("\n=== 消费类别分布 ===")
print(results['category_distribution'])
print("\n=== Top 5省份分布 ===")
print(results['province_distribution'].nlargest(5))
