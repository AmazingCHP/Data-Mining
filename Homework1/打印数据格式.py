# import os
# import json
# import pandas as pd
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
#
# # 设置显示选项
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
#
#
# def get_all_files(folder_path):
#     """获取文件夹下所有文件路径"""
#     file_paths = []
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             file_paths.append(file_path)
#     return file_paths
#
#
# def analyze_category_distribution(files):
#     """分析购买历史中的category分布"""
#     category_counts = pd.Series(dtype='int')
#
#     for file in files:
#         try:
#             # 读取Parquet文件
#             df = pd.read_parquet(file, engine='pyarrow')
#
#             # 检查是否有purchase_history列
#             if 'purchase_history' not in df.columns:
#                 continue
#
#             # 解析JSON并提取category
#             for history in df['purchase_history']:
#                 try:
#                     # 解析JSON字符串
#                     history_dict = json.loads(history) if isinstance(history, str) else history
#                     category = history_dict.get('category')
#                     if category:
#                         # 统计category出现次数
#                         if category in category_counts:
#                             category_counts[category] += 1
#                         else:
#                             category_counts[category] = 1
#                 except json.JSONDecodeError as e:
#                     print(f"JSON解析错误 (文件: {file}): {e}")
#                     continue
#                 except AttributeError as e:
#                     print(f"数据格式错误 (文件: {file}): {e}")
#                     continue
#
#         except Exception as e:
#             print(f"处理文件出错 {file}: {e}")
#             continue
#
#     if not category_counts.empty:
#         # 绘制柱状图
#         plt.figure(figsize=(12, 6))
#         ax = category_counts.sort_values(ascending=False).plot(
#             kind='bar',
#             color='skyblue',
#             edgecolor='black'
#         )
#
#         # 设置图表标题和标签
#         ax.set_title('Purchase Category Distribution', fontsize=14)
#         ax.set_xlabel('Category', fontsize=12)
#         ax.set_ylabel('Count', fontsize=12)
#         plt.xticks(rotation=45, ha='right')
#
#         # 添加数值标签
#         for p in ax.patches:
#             ax.annotate(
#                 f"{int(p.get_height())}",
#                 (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center', va='center',
#                 xytext=(0, 5),
#                 textcoords='offset points'
#             )
#
#         plt.tight_layout()
#         plt.show()
#
#         return category_counts.sort_values(ascending=False)
#     else:
#         print("未找到有效的category数据")
#         return None
#
#
# if __name__ == "__main__":
#     # 指定数据文件夹路径
#     data_folder = '10G_data'
#
#     print("当前字体:", plt.rcParams['font.sans-serif'])
#     plt.plot([1, '泰国'], label='测试中文')
#     plt.legend()
#     plt.show()
#
#     # 获取所有文件
#     files = get_all_files(data_folder)
#
#     if files:
#         print(f"共找到 {len(files)} 个文件")
#         result = analyze_category_distribution(files)
#
#         if result is not None:
#             print("\n品类分布统计:")
#             print(result.to_string())
#     else:
#         print("指定目录下未找到任何文件")

import os
import pandas as pd

# 设置显示选项以展开所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def get_all_files(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


string = '10G_data_new'

print(get_all_files(string))
for file in get_all_files(string):
    # 读取 Parquet 文件
    df = pd.read_parquet(file, engine='pyarrow')
    print(file)
    print(df.dtypes)

    # 输出前几行数据
    print(df.head(10))

    # 打印第一行信息
    print("第一行信息：")
    print(df.iloc[0])

    break  # 只查看第一个文件
