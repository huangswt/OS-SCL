import pandas as pd
import os
from tqdm import tqdm
name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']

for name in name_list:
    # 读取CSV文件
    df = pd.read_csv(f'{name}.csv', header=None)
    df.columns = ['filename', 'dummy', 'label']

    # 创建映射字典
    label_mapping = {row['filename']: row['label'] for idx, row in df.iterrows()}

    # 获取文件夹路径
    folder_path = f'{name}/test/'

    # 遍历目录中的文件
    file_list = os.listdir(folder_path)

    # 使用 tqdm 显示进度条
    for filename in tqdm(file_list, desc=f"Processing Machine {name}"):
        if filename in label_mapping:
            # 获取标签
            label = label_mapping[filename]
            # 构造新的文件名
            if label == 0:
                new_filename = 'normal_' + filename
            else:
                new_filename = 'anomaly_' + filename
            # 构造完整路径
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)
            # 重命名文件
            os.rename(old_file, new_file)

print("重命名完成。")
