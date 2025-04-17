import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体（Linux）
myfont = font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['axes.unicode_minus'] = False


def plot_gender_distribution_by_country(input_folder, output_image_path):
    all_data = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.parquet'):
            file_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_parquet(file_path, columns=["country", "gender"])
                all_data.append(df)
                print(f"✅ Processed: {filename}")
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

    if not all_data:
        print("⚠️ No data collected.")
        return

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)

    # 删除空值
    combined_df = combined_df.dropna(subset=["country", "gender"])

    # 统计每个国家各性别数量
    gender_counts = combined_df.groupby(["country", "gender"]).size().unstack(fill_value=0)

    # 绘制堆叠柱状图
    plt.figure(figsize=(16, 8))
    bottom = None
    for gender in gender_counts.columns:
        plt.bar(gender_counts.index, gender_counts[gender], label=gender, bottom=bottom)
        if bottom is None:
            bottom = gender_counts[gender]
        else:
            bottom += gender_counts[gender]

    plt.xlabel("国家", fontproperties=myfont)
    plt.ylabel("用户数量", fontproperties=myfont)
    plt.title("各国性别分布（堆叠柱状图）", fontproperties=myfont)
    plt.xticks(rotation=45, fontproperties=myfont)
    plt.legend(title="性别", prop=myfont)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path)
    print(f"📊 Gender distribution chart saved to: {output_image_path}")
    plt.close()


if __name__ == '__main__':
    input_folder = "10processed_data"
    output_image_path = os.path.join("pictures", "gender_distribution_by_country.png")
    plot_gender_distribution_by_country(input_folder, output_image_path)
