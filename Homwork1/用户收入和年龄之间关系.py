import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 设置中文字体（适用于Linux）
myfont = font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['axes.unicode_minus'] = False


def plot_income_age_heatmap(input_folder, output_image_path):
    all_data = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.parquet'):
            file_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_parquet(file_path, columns=["age", "income"])
                all_data.append(df)
                print(f"✅ Processed: {filename}")
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

    if not all_data:
        print("⚠️ No data collected.")
        return

    # 合并数据并去除空值
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna(subset=["age", "income"])

    # 年龄和收入转换为整数（如果不是）
    combined_df["age"] = combined_df["age"].astype(int)
    combined_df["income"] = combined_df["income"].astype(int)

    # 统计每个收入-年龄组合的数量
    heatmap_data = combined_df.groupby(["age", "income"]).size().unstack(fill_value=0)

    # 绘制热力图
    plt.figure(figsize=(18, 10))
    sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.5, linecolor='gray')

    plt.xlabel("收入", fontproperties=myfont)
    plt.ylabel("年龄", fontproperties=myfont)
    plt.title("用户收入与年龄之间的关系热力图", fontproperties=myfont)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path)
    print(f"📊 Income vs Age heatmap saved to: {output_image_path}")
    plt.close()


if __name__ == '__main__':
    input_folder = "10processed_data"
    output_image_path = os.path.join("pictures", "income_vs_age_heatmap.png")
    plot_income_age_heatmap(input_folder, output_image_path)
