import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


def plot_income_by_gender(input_folder, output_image_path):
    gender_income_data = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.parquet'):
            file_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_parquet(file_path, columns=["gender", "income"])
                gender_income_data.append(df)
                print(f"✅ Processed: {filename}")
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

    if not gender_income_data:
        print("⚠️ No data collected.")
        return

    combined_df = pd.concat(gender_income_data, ignore_index=True)
    combined_df = combined_df.dropna(subset=["gender", "income"])

    avg_income_by_gender = combined_df.groupby("gender")["income"].mean().sort_values(ascending=False)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.bar(avg_income_by_gender.index, avg_income_by_gender.values, color='coral')
    plt.xlabel("Gender")
    plt.ylabel("Average Income")
    plt.title("Average Income by Gender")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path)
    print(f"📊 Bar chart saved to: {output_image_path}")
    plt.close()


if __name__ == '__main__':
    input_folder = "10processed_data"
    output_image_path = os.path.join("pictures", "income_by_gender.png")
    plot_income_by_gender(input_folder, output_image_path)
