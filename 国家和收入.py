import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def process_all_files(folder_path):
    """处理文件夹下所有Parquet文件"""
    all_data = pd.DataFrame()

    # 遍历文件夹
    for file in os.listdir(folder_path):
        if file.endswith('.parquet'):
            file_path = os.path.join(folder_path, file)
            try:
                # 读取单个文件
                df = pd.read_parquet(file_path)
                # 筛选有效收入数据 (0 < income <= 1000000)
                df = df[(df['income'] > 0) & (df['income'] <= 1000000)]
                all_data = pd.concat([all_data, df])
                print(f"已处理文件: {file}")
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")
                continue

    return all_data


def analyze_income_by_country(df):
    """分析国家与收入关系"""
    if df.empty:
        return None

    # 将收入分为10档
    income_bins = np.linspace(0, 1000000, 11)
    income_labels = [f"{int(income_bins[i]) / 1000}k-{int(income_bins[i + 1]) / 1000}k"
                     for i in range(len(income_bins) - 1)]
    df['income_range'] = pd.cut(df['income'], bins=income_bins, labels=income_labels)

    # 按国家和收入档统计
    country_income = df.groupby(['country', 'income_range']).size().unstack().fillna(0)

    # 取用户数最多的前15个国家
    top_countries = df['country'].value_counts().head(15).index
    country_income = country_income.loc[top_countries]

    return country_income


# 主程序
if __name__ == "__main__":
    folder_path = "10G_data"

    if os.path.exists(folder_path):
        print(f"开始分析文件夹: {folder_path}")
        df = process_all_files(folder_path)

        if not df.empty:
            country_income = analyze_income_by_country(df)

            if country_income is not None:
                # 绘制热力图
                plt.figure(figsize=(16, 10))
                plt.imshow(country_income, cmap='YlOrRd', aspect='auto')

                # 设置坐标轴
                plt.xticks(np.arange(len(country_income.columns)), country_income.columns, rotation=45, ha='right')
                plt.yticks(np.arange(len(country_income.index)), country_income.index)
                plt.colorbar(label='用户数量')

                # 添加数值标签
                for i in range(len(country_income.index)):
                    for j in range(len(country_income.columns)):
                        count = country_income.iloc[i, j]
                        if count > 0:
                            plt.text(j, i, f"{int(count)}",
                                     ha='center', va='center', color='black', fontsize=8)

                plt.title('各国收入分布热力图 (Top 15国家)', fontsize=16)
                plt.xlabel('收入区间 (元)', fontsize=12)
                plt.ylabel('国家', fontsize=12)
                plt.tight_layout()
                plt.show()

                # 输出统计结果
                print("\n各国收入分布统计 (Top 15):")
                print(country_income)
            else:
                print("未能生成有效的收入分布数据")
        else:
            print("未找到有效数据")
    else:
        print(f"文件夹不存在: {folder_path}")
