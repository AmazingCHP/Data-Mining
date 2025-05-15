import pandas as pd
import json
import os
from tqdm import tqdm
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from datetime import datetime
from 子类 import product_dict

# 设置显示
pd.set_option('display.max_columns', None)
tqdm.pandas(desc="添加月份信息")

# 1. 读取数据并抽样
folder_path = '30G_data_new'
dfs = []

print("开始读取 parquet 文件并抽样...")
for filename in os.listdir(folder_path):
    if filename.endswith('.parquet'):
        print(f"读取: {filename}")
        df = pd.read_parquet(os.path.join(folder_path, filename), columns=['purchase_history'])
        sampled_df = df.sample(frac=0.0033, random_state=42)
        dfs.append(sampled_df)

user_df = pd.concat(dfs, ignore_index=True)
print(f"合并后共 {len(user_df)} 条记录")

# 2. 加载商品目录
with open("product_catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)
id_to_category = {item["id"]: item["category"] for item in catalog["products"]}


# 3. 月份提取函数
def get_month_str(date_str):
    try:
        month = datetime.strptime(date_str, "%Y-%m-%d").month
        return f"{month}月"
    except:
        return None


# 4. 提取大类 + 月份信息
def extract_items_with_month(purchase_str):
    try:
        record = json.loads(purchase_str)
        purchase_date = record.get("purchase_date", "")
        month = get_month_str(purchase_date)
        item_ids = [item["id"] for item in record.get("items", [])]
        categories = [id_to_category.get(i, None) for i in item_ids]
        main_categories = [product_dict.get(c, c) for c in categories if c]
        if month:
            main_categories.append(month)
        return list(set(main_categories))
    except:
        return []


# 5. 应用函数生成事务
print("提取每条订单的商品大类 + 月份信息...")
user_df["transaction_items"] = user_df["purchase_history"].progress_apply(extract_items_with_month)
transactions = user_df["transaction_items"].tolist()
print(transactions[:100])

# 6. One-hot 编码
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 7. Apriori 挖掘
print("执行 Apriori 挖掘...")
frequent_itemsets = apriori(df, min_support=0.005, use_colnames=True)

# 8. 生成关联规则
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)

# 9. 保存结果
frequent_itemsets.to_csv("月度频繁项集.csv", index=False, encoding="utf-8-sig")
rules.to_csv("月度关联规则.csv", index=False, encoding="utf-8-sig")

print("✅ 完成！月度频繁项集和关联规则已保存。")
