import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置字体（适配中文系统可省略）
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK', 'Microsoft YaHei', 'SimHei']  # 可以根据系统选择相应的字体
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块


def plot_payment_status_heatmap(input_folder, output_image_path):
    # 存储所有数据
    category_payment_data = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.parquet'):
            file_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_parquet(file_path, columns=["categories", "payment_status"])
                category_payment_data.append(df)
                print(f"✅ Processed: {filename}")
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")
            break

    if not category_payment_data:
        print("⚠️ No data collected.")
        return

    # 合并所有数据
    combined_df = pd.concat(category_payment_data, ignore_index=True)

    # 删除空值
    combined_df = combined_df.dropna(subset=["categories", "payment_status"])

    # 计算每个类别中不同支付状态的计数
    payment_counts = combined_df.groupby(["categories", "payment_status"]).size().unstack(fill_value=0)

    # 计算比例
    payment_ratio = payment_counts.div(payment_counts.sum(axis=1), axis=0)

    # 绘制热力图
    plt.figure(figsize=(14, 8))
    sns.heatmap(payment_ratio, annot=True, fmt=".2%", cmap="Blues", cbar_kws={'label': 'Percentage'},
                linewidths=0.5, linecolor='gray', xticklabels=payment_ratio.columns, yticklabels=payment_ratio.index)

    plt.xlabel("Payment Status")
    plt.ylabel("Item Category")
    plt.title("Payment Status Distribution by Item Category")
    plt.tight_layout()

    # 保存图像
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path)
    print(f"📊 Heatmap saved to: {output_image_path}")
    plt.close()


if __name__ == '__main__':
    input_folder = "10processed_data"
    output_image_path = os.path.join("pictures", "payment_status_heatmap.png")
    plot_payment_status_heatmap(input_folder, output_image_path)
