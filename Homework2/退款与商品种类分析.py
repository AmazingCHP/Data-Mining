import pandas as pd
import json
import os
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from tqdm import tqdm
from 子类 import product_dict

# 设置显示选项
pd.set_option('display.max_columns', None)

# tqdm 初始化
tqdm.pandas(desc="处理退款订单")

# 1. 读取 parquet 文件并抽样
folder_path = '30G_data_new'
dfs = []

print("开始读取 parquet 文件并进行抽样...")
for filename in os.listdir(folder_path):
    if filename.endswith('.parquet'):
        print(f"读取: {filename}")
        df = pd.read_parquet(os.path.join(folder_path, filename), columns=['purchase_history'])
        sampled_df = df.sample(frac=0.033, random_state=42)  # 八分之一抽样
        dfs.append(sampled_df)

user_df = pd.concat(dfs, ignore_index=True)
print(f"合并后总记录数: {len(user_df)}")

# 2. 读取商品目录
with open('product_catalog.json', 'r', encoding='utf-8') as f:
    catalog = json.load(f)
id_to_category = {item["id"]: item["category"] for item in catalog["products"]}


# 3. 过滤退款订单，并提取大类组合
def extract_refund_categories_with_tag(purchase_str):
    try:
        record = json.loads(purchase_str)
        status = record.get("payment_status", "")
        if status not in ["已退款", "部分退款"]:
            return None  # 非退款订单跳过
        item_ids = [item["id"] for item in record.get("items", [])]
        categories = [id_to_category.get(i, None) for i in item_ids]
        main_categories = [product_dict.get(c, c) for c in categories if c]

        if not main_categories:
            return None

        # 添加标签项
        main_categories.append(status)
        return list(set(main_categories))  # 去重
    except Exception as e:
        return None


print("筛选退款订单并提取商品大类组合（含退款标签）...")
user_df["refund_categories"] = user_df["purchase_history"].progress_apply(extract_refund_categories_with_tag)

# 4. 清理并准备事务数据
transactions = user_df["refund_categories"].dropna().tolist()
print(f"退款订单数（含标签）: {len(transactions)}")
print(transactions[:100])

# 5. One-hot 编码
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 6. 使用 Apriori 算法
print("开始 Apriori 挖掘...")
frequent_itemsets = apriori(df, min_support=0.005, use_colnames=True)
frequent_itemsets.to_csv("refund_related_frequent.csv", index=False, encoding='utf-8-sig')

# 7. 生成关联规则
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)

# 8. 保存结果
rules.to_csv("refund_related_rules.csv", index=False, encoding='utf-8-sig')
print("挖掘完成，结果保存为 refund_related_rules.csv")
