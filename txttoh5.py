import numpy as np
import h5py
import os
import glob
import laspy
def convert_txt_to_h5(txt_file, h5_file):
    # 读取TXT文件
    data = np.loadtxt(txt_file)
    print(data.shape)
    # 确保数据分为多个2048个点的片段
    num_points = 1024
    num_features = 6
    if data.shape[1] < num_features + 1:  # 确保数据至少有3个特征和1个标签
        raise ValueError("Data must have at least 3 features and 1 label")

    # 分离特征和标签
    features = data[:, :num_features]
    labels = data[:, -1].astype('int64')
    #labels = np.full(features.shape[0], 1, dtype='int64')
    # 计算片段数量
    num_samples = len(features) // num_points
    # remainder = len(features) % num_points
    # if remainder != 0:
    #     num_samples += 1

    # 初始化存储数据
    features_padded = np.zeros((num_samples, num_points, num_features))
    labels_padded = np.zeros((num_samples,num_points), dtype='int64')

    for i in range(num_samples):
        start_idx = i * num_points
        end_idx = min((i + 1) * num_points, len(features))
        current_features = features[start_idx:end_idx]
        current_labels = labels[start_idx:end_idx]
        if len(current_features) < num_points:
            repeat_count = num_points - len(current_features)
            # 重复当前批次的点来填充
            repeat_features = np.tile(current_features, (repeat_count // len(current_features) + 1, 1))
            repeat_labels = np.tile(current_labels, (repeat_count // len(current_labels) + 1))

            # 取前repeat_count个点
            current_features = np.concatenate([current_features, repeat_features[:repeat_count]], axis=0)
            current_labels = np.concatenate([current_labels, repeat_labels[:repeat_count]], axis=0)

        features_padded[i] = current_features
        labels_padded[i] = current_labels
    print(features_padded.shape)
    print(labels_padded.shape)
    # 创建HDF5文件并保存数据
    with h5py.File(h5_file, 'w') as hf:
        hf.create_dataset('data', data=features_padded)
        hf.create_dataset('label', data=labels_padded)

    print(f"Converted {txt_file} to {h5_file}")

def convert_all_txt_to_h5(input_dir, output_dir):
    # 获取所有TXT文件
    txt_files = glob.glob(os.path.join(input_dir, '*.txt'))

    h5_files = []
    for txt_file in txt_files:
        base_name = os.path.basename(txt_file).replace('.txt', '.h5')
        h5_file = os.path.join(output_dir, base_name)
        convert_txt_to_h5(txt_file, h5_file)
        h5_files.append(h5_file)

    return h5_files

def split_h5_files(h5_files, output_dir):
    # 随机打乱HDF5文件列表
    np.random.shuffle(h5_files)

    # 计算分割索引
    split_index = int(len(h5_files) * 1)

    # 划分训练集和验证集
    train_files = h5_files[:split_index]
    val_files = h5_files[split_index:]

    # 移动文件并重命名
    for h5_file in train_files:
        new_name = os.path.join(output_dir, f'train_{os.path.basename(h5_file)}')
        os.rename(h5_file, new_name)

    for h5_file in val_files:
        new_name = os.path.join(output_dir, f'test_{os.path.basename(h5_file)}')
        os.rename(h5_file, new_name)

    print(f"Split files into train and test sets.")
def convert_las_to_h5(las_file, h5_file, num_points=1024, num_features=6):
    # 打开 LAS 文件
    las = laspy.read(las_file)

    # 提取点云坐标和 RGB 颜色（或其他特征）
    x = las.x
    y = las.y
    z = las.z
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        r = las.red
        g = las.green
        b = las.blue
    else:
        # 如果没有 RGB 信息，用默认值填充
        r, g, b = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)

    # 合并为特征数组
    features = np.vstack([x, y, z, r, g, b]).T

    # 创建虚拟标签（如果没有实际标签）
    labels = np.full(features.shape[0], 1, dtype='int64')  # 默认标签全为 1

    # 确保数据分为多个 num_points 的片段
    num_samples = len(features) // num_points
    features_padded = np.zeros((num_samples, num_points, num_features))
    labels_padded = np.zeros((num_samples, num_points), dtype='int64')

    for i in range(num_samples):
        start_idx = i * num_points
        end_idx = min((i + 1) * num_points, len(features))
        current_features = features[start_idx:end_idx]
        current_labels = labels[start_idx:end_idx]
        if len(current_features) < num_points:
            repeat_count = num_points - len(current_features)
            # 用当前批次的点填充
            repeat_features = np.tile(current_features, (repeat_count // len(current_features) + 1, 1))
            repeat_labels = np.tile(current_labels, (repeat_count // len(current_labels) + 1))

            # 截取填充的点
            current_features = np.concatenate([current_features, repeat_features[:repeat_count]], axis=0)
            current_labels = np.concatenate([current_labels, repeat_labels[:repeat_count]], axis=0)

        features_padded[i] = current_features
        labels_padded[i] = current_labels

    # 保存为 HDF5 文件
    with h5py.File(h5_file, 'w') as hf:
        hf.create_dataset('data', data=features_padded)
        hf.create_dataset('label', data=labels_padded)

    print(f"Converted {las_file} to {h5_file}")


def convert_all_las_to_h5(input_dir, output_dir, num_points=1024):
    # 获取所有 LAS 文件
    las_files = glob.glob(os.path.join(input_dir, '*.las'))

    h5_files = []
    for las_file in las_files:
        base_name = os.path.basename(las_file).replace('.las', '.h5')
        h5_file = os.path.join(output_dir, base_name)
        convert_las_to_h5(las_file, h5_file, num_points=num_points)
        h5_files.append(h5_file)

    return h5_files


def split_h5_files(h5_files, output_dir):
    # 随机打乱 HDF5 文件列表
    np.random.shuffle(h5_files)

    # 计算分割索引
    split_index = int(len(h5_files) * 0)  # 80% 用作训练集

    # 划分训练集和验证集
    train_files = h5_files[:split_index]
    val_files = h5_files[split_index:]

    # 移动文件并重命名
    for h5_file in train_files:
        new_name = os.path.join(output_dir, f'train_{os.path.basename(h5_file)}')
        os.rename(h5_file, new_name)

    for h5_file in val_files:
        new_name = os.path.join(output_dir, f'test_{os.path.basename(h5_file)}')
        os.rename(h5_file, new_name)

    print(f"Split files into train and test sets.")

# 示例使用
input_dir = "E:/lcy/dgcnn/data/coordinate_correction/bridge2_proceed_las"  # 替换为你的TXT文件目录
output_dir = "E:/lcy/dgcnn/data/coordinate_correction/bridge2_seg_1024_1"  # 替换为输出HDF5文件目录

os.makedirs(output_dir, exist_ok=True)

# 首先转换所有TXT文件
h5_files = convert_all_txt_to_h5(input_dir, output_dir)

# 然后分割HDF5文件
#split_h5_files(h5_files, output_dir)

# 首先转换所有 LAS 文件
#h5_files = convert_all_las_to_h5(input_dir, output_dir)

# 然后分割 HDF5 文件
#split_h5_files(h5_files, output_dir)

