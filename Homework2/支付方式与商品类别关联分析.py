import pandas as pd
import json
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from tqdm import tqdm
import glob
import os
from 子类 import product_dict

# 🔹 1. 读取商品目录 JSON，构建映射表
print("🔹 正在加载商品目录...")
with open("product_catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)
id_to_category = {item["id"]: item["category"] for item in catalog["products"]}
print("✅ 商品目录加载完成，共计商品数：", len(id_to_category))


# 🔹 2. 定义特征提取函数
def extract_features(row):
    try:
        record = json.loads(row["purchase_history"])
        features_list = []
        items = record.get("items", [])
        payment_method = record.get("payment_method")
        payment_feature = f"{payment_method}" if payment_method else None
        for item in items:
            feature = []
            product_id = item.get("id")
            category = id_to_category.get(product_id)
            if category:
                feature.append(product_dict.get(category, category))
            if payment_feature:
                feature.append(payment_feature)
            if feature:
                features_list.append(feature)
        return features_list
    except Exception:
        return []


# 🔹 3. 加载并采样每个 parquet 文件
parquet_files = glob.glob(os.path.join("30G_data_new", "*.parquet"))
sampled_list = []
print(f"🔹 共找到 {len(parquet_files)} 个数据文件，开始逐个读取并采样八分之一...")

for i, file in enumerate(parquet_files):
    print(f"📂 [{i + 1}/{len(parquet_files)}] 正在处理文件：{file}")
    df = pd.read_parquet(file, columns=["purchase_history"])
    df_sample = df.sample(frac=0.033, random_state=42)
    sampled_list.append(df_sample)
    print(f"   ✅ 完成采样，原始数据 {len(df)} 条，采样后 {len(df_sample)} 条")

# 🔹 4. 合并所有采样数据
df_all = pd.concat(sampled_list, ignore_index=True)
print(f"🔹 所有采样合并完成，总计记录数：{len(df_all)}")

# 🔹 5. 提取事务特征
print("🔹 正在提取事务特征（带进度条）...")
tqdm.pandas(desc="⏳ 提取中")
df_all["features"] = df_all.progress_apply(extract_features, axis=1)
print("✅ 特征提取完成")

# 🔹 6. 展平特征列表
print("🔹 正在展平特征列表...")
data = [feat for sublist in df_all["features"] for feat in sublist if feat]
print(f"✅ 特征展平完成，共计事务数：{len(data)}")
if not data:
    print("⚠️ 没有有效事务数据，终止分析")
    exit()

# 🔹 7. TransactionEncoder 编码
print("🔹 正在进行 TransactionEncoder 编码...")
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
print("✅ 编码完成，生成的维度数：", len(df_encoded.columns))

# 🔹 8. 执行 Apriori 分析
print("🔹 正在执行 Apriori 算法...")
frequent_itemsets = apriori(df_encoded, min_support=0.002, use_colnames=True)
frequent_itemsets.to_csv("频繁项集_采样合并.csv", index=False, encoding="utf-8-sig")
print(f"✅ 频繁项集挖掘完成，共找到 {len(frequent_itemsets)} 个频繁项集")

# 🔹 9. 生成关联规则
print("🔹 正在生成关联规则...")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0)
rules = rules.sort_values(by='lift', ascending=False)
rules.to_csv("关联规则_采样合并.csv", index=False, encoding="utf-8-sig")
print(f"✅ 关联规则生成完成，共计规则数：{len(rules)}")

# 🔹 10. 打印前几条结果
print("\n📊 前5条频繁项集：")
print(frequent_itemsets.head())

print("\n🔗 前5条关联规则：")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
