import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from matplotlib import font_manager

# 设置中文字体（适用于Linux）
myfont = font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['axes.unicode_minus'] = False


def plot_income_vs_spending_with_clustering(input_folder, output_image_path):
    all_data = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.parquet'):
            file_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_parquet(file_path, columns=["income", "spending_amount"])
                all_data.append(df)
                print(f"✅ Processed: {filename}")
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

    if not all_data:
        print("⚠️ No data collected.")
        return

    # 合并数据并去除空值
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna(subset=["income", "spending_amount"])

    # 数据归一化
    scaler = MinMaxScaler()
    combined_df[["income", "spending_amount"]] = scaler.fit_transform(combined_df[["income", "spending_amount"]])

    # K-means 聚类
    kmeans = KMeans(n_clusters=4, random_state=42)
    combined_df['cluster'] = kmeans.fit_predict(combined_df[["income", "spending_amount"]])

    # 绘制散点图
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x="income", y="spending_amount", hue="cluster", data=combined_df, palette="Set1", s=100, alpha=0.7)

    plt.xlabel("收入 (归一化)", fontproperties=myfont)
    plt.ylabel("消费金额 (归一化)", fontproperties=myfont)
    plt.title("收入与消费金额的关系及K-means聚类结果", fontproperties=myfont)
    plt.legend(title="Cluster", loc="upper right")
    plt.tight_layout()

    # 保存图像
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path)
    print(f"📊 Income vs Spending with Clustering saved to: {output_image_path}")
    plt.close()


if __name__ == '__main__':
    input_folder = "10processed_data"
    output_image_path = os.path.join("pictures", "income_vs_spending_clustering.png")
    plot_income_vs_spending_with_clustering(input_folder, output_image_path)
