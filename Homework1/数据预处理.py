import os
import pandas as pd
import json
import re
import gc
from datetime import datetime
from pandas import json_normalize
import pyarrow.parquet as pq


def clean_and_process(df):
    df['last_login'] = pd.to_datetime(df['last_login'], errors='coerce').dt.tz_localize(None)
    df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')

    try:
        df['purchase_history'] = df['purchase_history'].apply(json.loads)
        purchase_df = json_normalize(df['purchase_history'])
        df = pd.concat([df.drop('purchase_history', axis=1), purchase_df], axis=1)
        df['items_count'] = df['items'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df.drop(columns=['items'], inplace=True)
    except Exception as e:
        print(f"购买历史字段处理失败: {e}")
        df['items_count'] = 0

    try:
        df['login_history'] = df['login_history'].apply(json.loads)
        df['login_count'] = df['login_history'].apply(lambda x: x.get('login_count', 0))
        df['first_login'] = pd.to_datetime(df['login_history'].apply(lambda x: x.get('first_login')))
        df['device_count'] = df['login_history'].apply(lambda x: len(set(x.get('devices', []))))
        df['location_count'] = df['login_history'].apply(lambda x: len(set(x.get('locations', []))))
        df.drop(columns=['login_history'], inplace=True)
    except Exception as e:
        print(f"登录历史字段处理失败: {e}")
        df['login_count'] = 0
        df['device_count'] = 0
        df['location_count'] = 0

    df['province'] = df['address'].apply(
        lambda x: re.search(r'(.*?省)', x).group(1) if isinstance(x, str) and '省' in x else '其他'
    )

    df['gender'] = df['gender'].fillna('未知')
    df['age'] = df['age'].fillna(df['age'].median())
    df['income'] = df['income'].fillna(df['income'].median())

    df = df[(df['age'] > 0) & (df['age'] < 120)]
    Q1 = df['income'].quantile(0.25)
    Q3 = df['income'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['income'] >= (Q1 - 1.5 * IQR)) & (df['income'] <= (Q3 + 1.5 * IQR))]

    df['days_since_registration'] = (df['last_login'] - df['registration_date']).dt.days
    df['login_hour'] = df['last_login'].dt.hour
    df['login_dayofweek'] = df['last_login'].dt.dayofweek
    df['is_weekend'] = df['login_dayofweek'].isin([5, 6]).astype(int)

    return df


def process_large_parquet_file(input_file, output_folder):
    pf = pq.ParquetFile(input_file)
    total_row_groups = pf.num_row_groups
    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    for i in range(total_row_groups):
        try:
            batch_table = pf.read_row_group(i).to_pandas()
            cleaned_batch = clean_and_process(batch_table)

            output_file = os.path.join(output_folder, f"{base_filename}_part{i}.parquet")
            cleaned_batch.to_parquet(output_file, index=False)
            print(f"✅ 成功处理 row_group {i}，保存为：{output_file}")

            del batch_table, cleaned_batch
            gc.collect()
        except Exception as e:
            print(f"❌ 处理 row_group {i} 失败：{e}")


def main():
    input_dir = '10G_data_new'
    output_dir = '10processed_data'
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith('.parquet'):
            input_file_path = os.path.join(input_dir, file)
            print(f"正在处理文件：{input_file_path}")
            try:
                process_large_parquet_file(input_file_path, output_dir)
            except Exception as e:
                print(f"  !!! 文件处理失败：{input_file_path}，错误：{e}")


if __name__ == '__main__':
    main()
