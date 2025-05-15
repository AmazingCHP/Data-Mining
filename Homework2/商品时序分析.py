import pandas as pd
import json
import os
from tqdm import tqdm
from collections import Counter
from 子类 import product_dict  # 用于大类映射

# 显示所有列
pd.set_option('display.max_columns', None)
tqdm.pandas(desc="解析购买顺序")

# 读取商品目录
with open("product_catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)
id_to_category = {item["id"]: item["category"] for item in catalog["products"]}


# 函数：将购买记录转为有序大类序列对
def extract_category_pairs(purchase_str):
    try:
        record = json.loads(purchase_str)
        item_ids = [item["id"] for item in record.get("items", [])]
        categories = [product_dict.get(id_to_category.get(i, None), None) for i in item_ids]
        categories = [c for c in categories if c is not None]

        pairs = []
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                if categories[i] != categories[j]:  # 前后不同
                    pairs.append((categories[i], categories[j]))
        return pairs
    except:
        return []


# 读取和抽样 parquet 文件
folder_path = '30G_data_new'
all_pairs = []

print("读取 parquet 并提取购买顺序对...")
for filename in os.listdir(folder_path):
    if filename.endswith(".parquet"):
        print(f"读取: {filename}")
        df = pd.read_parquet(os.path.join(folder_path, filename), columns=['purchase_history'])
        sampled_df = df.sample(frac=0.033, random_state=42)
        sampled_df['pair_list'] = sampled_df['purchase_history'].progress_apply(extract_category_pairs)
        for row in sampled_df['pair_list']:
            all_pairs.extend(row)

# 统计频率
pair_counter = Counter(all_pairs)
total = sum(pair_counter.values())

# 构造结果 DataFrame
df_result = pd.DataFrame([
    {"from": k[0], "to": k[1], "count": v, "support": v / total}
    for k, v in pair_counter.items()
])
df_result = df_result.sort_values(by="count", ascending=False)

# 保存 CSV
df_result.to_csv("商品大类顺序关系.csv", index=False, encoding="utf-8-sig")
print("✅ 完成！顺序关系结果已保存：商品大类顺序关系.csv")
