import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def process_all_files(folder_path):
    """处理文件夹下所有Parquet文件"""
    all_gender_counts = pd.DataFrame()

    # 遍历文件夹
    for file in os.listdir(folder_path):
        if file.endswith('.parquet'):
            file_path = os.path.join(folder_path, file)
            try:
                # 读取单个文件
                df = pd.read_parquet(file_path)

                # 统计当前文件的性别分布
                gender_counts = df.groupby(['country', 'gender']).size().unstack()
                all_gender_counts = pd.concat([all_gender_counts, gender_counts])

                print(f"已处理文件: {file}")
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")
                continue

    # 合并所有文件的统计结果
    if not all_gender_counts.empty:
        # 按国家分组求和
        final_counts = all_gender_counts.groupby('country').sum()
        # 筛选前20国家
        top_countries = final_counts.sum(axis=1).sort_values(ascending=False).head(20).index
        return final_counts.loc[top_countries]
    else:
        return None


# 主程序
if __name__ == "__main__":
    folder_path = "10G_data"

    if os.path.exists(folder_path):
        print(f"开始分析文件夹: {folder_path}")
        gender_counts = process_all_files(folder_path)

        if gender_counts is not None:
            # 绘制图表
            plt.figure(figsize=(16, 8))
            bar_width = 0.35
            x = np.arange(len(gender_counts.index))

            # 绘制柱子
            plt.bar(x - bar_width / 2, gender_counts['男'], width=bar_width, label='男', color='#1f77b4')
            plt.bar(x + bar_width / 2, gender_counts['女'], width=bar_width, label='女', color='#ff7f0e')

            # 图表装饰
            plt.title('各国用户性别分布（Top 20国家 - 全量数据）', fontsize=16)
            plt.xlabel('国家', fontsize=12)
            plt.ylabel('用户数量', fontsize=12)
            plt.xticks(x, gender_counts.index, rotation=45, ha='right')
            plt.legend(title='性别', bbox_to_anchor=(1.05, 1))

            # 添加数值标签（自动调整位置）
            max_value = gender_counts.max().max()
            label_offset = max_value * 0.02  # 动态偏移量

            for i in x:
                male_val = gender_counts['男'].iloc[i]
                female_val = gender_counts['女'].iloc[i]
                if male_val > 0:
                    plt.text(i - bar_width / 2, male_val + label_offset, f"{male_val:,}",
                             ha='center', va='bottom', fontsize=9)
                if female_val > 0:
                    plt.text(i + bar_width / 2, female_val + label_offset, f"{female_val:,}",
                             ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.show()

            # 输出统计结果
            print("\n各国性别分布统计（Top 20）：")
            print(gender_counts)
        else:
            print("未找到有效数据")
    else:
        print(f"文件夹不存在: {folder_path}")
