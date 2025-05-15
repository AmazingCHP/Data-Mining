import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置字体（适配中文系统可省略）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_avg_price_by_category(input_folder, output_image_path):
    category_price_data = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.parquet'):
            file_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_parquet(file_path, columns=["payment_method", "avg_price"])
                category_price_data.append(df)
                print(f"✅ Processed: {filename}")
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

    if not category_price_data:
        print("⚠️ No data collected.")
        return

    # 合并所有数据
    combined_df = pd.concat(category_price_data, ignore_index=True)

    # 删除空值
    combined_df = combined_df.dropna(subset=["payment_method", "avg_price"])

    # 按类别计算平均价格
    avg_price_by_category = combined_df.groupby("payment_method")["avg_price"].mean().sort_values(ascending=False)

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.bar(avg_price_by_category.index, avg_price_by_category.values, color='skyblue')
    plt.xlabel("Item Category")
    plt.ylabel("Average Price")
    plt.title("Average Price by Item Category")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图像
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path)
    print(f"📊 Bar chart saved to: {output_image_path}")
    plt.close()


if __name__ == '__main__':
    input_folder = "10processed_data"
    output_image_path = os.path.join("pictures", "avg_price_by_payment_method.png")
    plot_avg_price_by_category(input_folder, output_image_path)
