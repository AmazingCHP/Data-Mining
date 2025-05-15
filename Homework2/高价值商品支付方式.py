import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 加载商品目录，获取 id → price 映射
with open("product_catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

id_to_price = {item["id"]: item["price"] for item in catalog["products"]}

# 加载主数据集
df = pd.read_parquet("10G_data_new/part-00000.parquet")  # 或 CSV，根据你格式来改

# 初始化计数字典
from collections import Counter

payment_counter = Counter()

# 遍历每条购买记录
for _, row in df.iterrows():
    try:
        record = json.loads(row["purchase_history"])
        items = record.get("items", [])
        payment_method = record.get("payment_method")
        for item in items:
            pid = item.get("id")
            price = id_to_price.get(pid, 0)
            if price > 5000 and payment_method:
                payment_counter[payment_method] += 1
    except Exception:
        continue

# 转换为 DataFrame 方便绘图
payment_df = pd.DataFrame(payment_counter.items(), columns=["支付方式", "数量"])

# 对数据进行排序，数量由高到低
payment_df = payment_df.sort_values(by="数量", ascending=False)

# -------------------
# 条形图
plt.figure(figsize=(8, 5))
plt.barh(payment_df["支付方式"], payment_df["数量"], color="skyblue")
plt.title("高价值商品的支付方式（横向柱状图）")
plt.xlabel("数量")
plt.ylabel("支付方式")
plt.tight_layout()

plt.savefig("高价值商品的支付方式.png", dpi=300)

plt.show()
