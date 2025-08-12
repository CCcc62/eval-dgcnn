from data import load_my_data
import numpy as np
all_data, labels = load_my_data(partition='test', exp_name= 'seg_1024_1')
# 统计每个类别的点云数量
unique_labels, label_counts = np.unique(labels, return_counts=True)

# 打印每个类别的点云数量
print("Label counts:")
for label, count in zip(unique_labels, label_counts):
    print(f"Class {label}: {count} points")
