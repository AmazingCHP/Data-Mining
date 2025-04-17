import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置字体（适配中文系统可省略）
myfont = font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')  # 根据你的系统调整路径
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块


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
    bars = plt.bar(avg_price_by_category.index, avg_price_by_category.values, color='skyblue')
    plt.xlabel("支付方式", fontproperties=myfont)
    plt.ylabel("平均价格", fontproperties=myfont)
    plt.title("按支付方式计算的平均价格", fontproperties=myfont)
    plt.xticks(rotation=45, fontproperties=myfont)

    # 设置y轴范围（略低于最小值）
    min_price = avg_price_by_category.min()
    max_price = avg_price_by_category.max()
    y_min = min_price - (min_price * 0.05)  # 将y轴下限设置为略低于最小值的5%
    plt.ylim(y_min, max_price * 1.05)

    plt.tight_layout()

    # 保存图像
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path)
    print(f"📊 Bar chart saved to: {output_image_path}")
    plt.close()


if __name__ == '__main__':
    input_folder = "10processed_data"
    output_image_path = os.path.join("pictures", "avg_price_by_category.png")
    plot_avg_price_by_category(input_folder, output_image_path)
