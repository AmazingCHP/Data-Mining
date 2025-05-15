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
tqdm.pandas(desc="添加季度信息")

# 1. 读取数据并抽样
folder_path = '30G_data_new'
dfs = []

print("开始读取 parquet 文件并抽样...")
for filename in os.listdir(folder_path):
    if filename.endswith('.parquet'):
        print(f"读取: {filename}")
        df = pd.read_parquet(os.path.join(folder_path, filename), columns=['purchase_history'])
        sampled_df = df.sample(frac=0.033, random_state=42)
        dfs.append(sampled_df)

user_df = pd.concat(dfs, ignore_index=True)
print(f"合并后共 {len(user_df)} 条记录")

# 2. 加载商品目录
with open("product_catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)
id_to_category = {item["id"]: item["category"] for item in catalog["products"]}


# 3. 季度判断函数
def get_quarter_str(date_str):
    try:
        month = datetime.strptime(date_str, "%Y-%m-%d").month
        if 1 <= month <= 3:
            return "第一季度"
        elif 4 <= month <= 6:
            return "第二季度"
        elif 7 <= month <= 9:
            return "第三季度"
        else:
            return "第四季度"
    except:
        return None


# 4. 提取大类 + 季节信息
def extract_items_with_quarter(purchase_str):
    try:
        record = json.loads(purchase_str)
        purchase_date = record.get("purchase_date", "")
        quarter = get_quarter_str(purchase_date)
        item_ids = [item["id"] for item in record.get("items", [])]
        categories = [id_to_category.get(i, None) for i in item_ids]
        main_categories = [product_dict.get(c, c) for c in categories if c]
        if quarter:
            main_categories.append(quarter)
        return list(set(main_categories))  # 去重
    except:
        return []


# 5. 应用函数生成事务
print("提取每条订单的商品大类 + 季节信息...")
user_df["transaction_items"] = user_df["purchase_history"].progress_apply(extract_items_with_quarter)
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
frequent_itemsets.to_csv("3季度频繁.csv", index=False, encoding="utf-8-sig")
rules.to_csv("3季度关联.csv", index=False, encoding="utf-8-sig")

print("✅ 完成！频繁项集和关联规则已保存。")
