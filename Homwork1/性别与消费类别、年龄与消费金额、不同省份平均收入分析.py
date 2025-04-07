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

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 确保保存图片的文件夹存在
if not os.path.exists('pictures'):
    os.makedirs('pictures')


# 定义加载数据函数
def load_data(file_pattern, chunk_size=10):
    files = glob(file_pattern)
    for i in tqdm(range(0, len(files), chunk_size), desc="读取进度"):
        for file in files[i:i + chunk_size]:
            df = pd.read_parquet(file)
            yield df
            del df
            gc.collect()
        break

def gender_category_association(df):
    # 只处理 '男' 和 '女'，其他值设置为 NaN
    df['gender'] = df['gender'].apply(lambda x: x if x in ['男', '女'] else None)

    # 删除那些性别不是 '男' 或 '女' 的行
    df = df.dropna(subset=['gender'])

    # 将 '男' 转化为 0，'女' 转化为 1
    df.loc[:, 'gender'] = df['gender'].map({'男': 0, '女': 1})

    # 对 category 列进行独热编码，将其转化为多个二进制列
    df_combined = pd.get_dummies(df[['gender', 'category']])

    # 确保所有数据为布尔类型（0或1）
    df_combined = df_combined.astype(bool)

    # 使用 Apriori 算法进行频繁项集挖掘
    frequent_itemsets = apriori(df_combined, min_support=0.05, use_colnames=True)

    # 生成关联规则
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    return rules


def age_vs_spending(df):
    # 将年龄分段
    age_bins = [18, 25, 35, 45, 55, 65, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

    # 计算不同年龄段的平均消费金额
    age_spending = df.groupby('age_group')['average_price'].mean().reset_index()

    return age_spending


def province_vs_income(df):
    # 按省份分组计算平均收入
    province_income = df.groupby('province')['income'].mean().reset_index()

    # 排序并取Top 5
    top_provinces = province_income.nlargest(5, 'income')
    return top_provinces


# 执行数据分析
def analyze_data(file_pattern='processed_data/*.parquet'):
    """执行数据分析"""
    gender_category_rules = []
    age_spending_data = []
    province_income_data = []

    for df in load_data(file_pattern, chunk_size=10):
        # 1. 性别与消费类别关联规则
        gender_category_rules.append(gender_category_association(df))

        # 2. 年龄与消费金额关系
        age_spending_data.append(age_vs_spending(df))

        # 3. 地区与收入关系
        province_income_data.append(province_vs_income(df))

        del df
        gc.collect()

    # 处理关联规则结果
    gender_category_associations = pd.concat(gender_category_rules, ignore_index=True)

    # 处理年龄与消费金额分析结果
    age_spending_combined = pd.concat(age_spending_data, ignore_index=True)

    # 处理地区与收入分析结果
    province_income_combined = pd.concat(province_income_data, ignore_index=True)

    # 可视化分析结果 --------------------------------------------------

    # 1. 性别与消费类别的关联规则
    plt.figure(figsize=(12, 8))
    gender_category_associations.plot(kind='scatter', x='lift', y='confidence', c='support', cmap='coolwarm', s=100,
                                      alpha=0.6)
    plt.title('性别与消费类别的关联规则', fontsize=15)
    plt.xlabel('提升度 (Lift)', fontsize=12)
    plt.ylabel('置信度 (Confidence)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pictures/gender_category_association.png')  # 保存图片
    plt.close()

    # 2. 年龄与消费金额关系
    plt.figure(figsize=(12, 6))
    sns.barplot(x='age_group', y='average_price', data=age_spending_combined, palette='Blues_d')
    plt.title('不同年龄段的平均消费金额', fontsize=15)
    plt.xlabel('年龄段', fontsize=12)
    plt.ylabel('平均消费金额', fontsize=12)
    plt.tight_layout()
    plt.savefig('pictures/age_spending.png')  # 保存图片
    plt.close()

    # 3. 地区与收入关系
    plt.figure(figsize=(12, 6))
    sns.barplot(x='province', y='income', data=province_income_combined, palette='Blues_d')
    plt.title('不同省份的平均收入', fontsize=15)
    plt.xlabel('省份', fontsize=12)
    plt.ylabel('平均收入', fontsize=12)
    plt.tight_layout()
    plt.savefig('pictures/province_income.png')  # 保存图片
    plt.close()

    return {
        'gender_category_associations': gender_category_associations,
        'age_spending': age_spending_combined,
        'province_income': province_income_combined
    }


# 执行分析
results = analyze_data()

# 打印统计结果
print("=== 性别与消费类别的关联规则 ===")
print(results['gender_category_associations'].head())
print("\n=== 年龄与消费金额的关系 ===")
print(results['age_spending'])
print("\n=== 不同省份的平均收入 ===")
print(results['province_income'])
