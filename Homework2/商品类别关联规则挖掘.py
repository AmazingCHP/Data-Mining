import pandas as pd
import json
import os
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from tqdm import tqdm
from 子类 import product_dict

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 使用 tqdm 显示进度条
tqdm.pandas(desc="提取商品类别和大类")

# 1. 读取所有 parquet 文件并进行抽样
folder_path = '30G_data_new'
dfs = []

print("开始读取所有 parquet 文件并进行八分之一抽样...")
for filename in os.listdir(folder_path):
    if filename.endswith('.parquet'):
        file_path = os.path.join(folder_path, filename)
        print(f"正在读取并抽样文件: {filename}")
        try:
            df = pd.read_parquet(file_path, columns=['purchase_history'])
            sampled_df = df.sample(frac=0.0033, random_state=42)  # 抽样 1/8
            dfs.append(sampled_df)
        except Exception as e:
            print(f"读取或抽样失败: {filename}, 错误: {e}")

user_df = pd.concat(dfs, ignore_index=True)
print("所有文件抽样并合并完成，总样本数：", len(user_df))

# 2. 读取商品目录
print("开始读取商品目录...")
with open('product_catalog.json', 'r', encoding='utf-8') as f:
    product_catalog = json.load(f)

id_to_category = {item["id"]: item["category"] for item in product_catalog["products"]}
print("商品目录读取完成。")

# 3. 提取每个订单的商品类别，并根据 id_to_category 得到大类
print("提取商品类别和大类...")


def extract_categories(purchase_history_str):
    try:
        record = json.loads(purchase_history_str)
        item_ids = [item["id"] for item in record.get("items", [])]
        categories = [id_to_category.get(item_id, None) for item_id in item_ids]

        # 查找大类
        main_categories = [product_dict.get(category, category) for category in categories if category]
        return list(set(main_categories))
    except Exception as e:
        print(f"解析失败：{purchase_history_str[:50]}... 错误：{e}")
        return []


user_df["categories_list"] = user_df["purchase_history"].progress_apply(extract_categories)
transactions = user_df["categories_list"].tolist()
print("商品类别提取完成。")

# 输出部分数据用于调试
print(f"提取的前100条交易类别：\n{transactions[:100]}")

# 4. One-hot 编码
print("开始进行 One-hot 编码...")
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
print("One-hot 编码完成。")

# 5. 使用 FP-Growth 挖掘频繁项集
print("开始挖掘频繁项集...")
frequent_itemsets = fpgrowth(df, min_support=0.005, use_colnames=True)
frequent_itemsets.to_csv("all_frequent.csv", index=False, encoding='utf-8-sig')
print("频繁项集挖掘完成。")

# 6. 生成关联规则
print("开始生成关联规则...")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.0)
print("关联规则生成完成。")

# 7. 将规则保存到文件
print("开始将规则保存到文件...")
rules.to_csv("all_rules.csv", index=False, encoding='utf-8-sig')
print("规则已保存到 all_rules.csv 文件。")

print("处理完毕，关联规则已保存。")
