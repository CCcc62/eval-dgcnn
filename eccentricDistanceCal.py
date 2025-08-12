import numpy as np
from sklearn.cluster import KMeans
import open3d as o3d
import matplotlib.pyplot as plt
import os
import time

def load_and_process_txt(file_path):
    """
    读取txt文件并处理数据
    文件格式假设为：x, y, z, label（每行一个点）
    """
    data = np.loadtxt(file_path)
    points = data[:, :3]  # 获取坐标 (x, y, z)
    labels = data[:, -1]   # 获取标签
    return points, labels

def get_sidewalk_boundaries(points, labels):
    """
    获取人行道边界点
    每个断面按 x 排序，再按 z 排序，找到每个断面 z 值最小且小于0的点作为左边界，
    找到每个断面 z 值最大且大于0的点作为右边界
    """
    sidewalk_points = points[labels == 1]
    x_values = np.unique(sidewalk_points[:, 0])
    left_boundaries = []
    right_boundaries = []

    for x in x_values:
        current_points = sidewalk_points[sidewalk_points[:, 0] == x]
        sorted_points = current_points[np.argsort(current_points[:, 2])]  # 按 z 排序

        # 获取左边界（z最小且小于0的点）
        left_point = sorted_points[sorted_points[:, 2] < 0]
        if len(left_point) > 0:
            left_boundaries.append(left_point[0])  # 左边界是最小的 z 值点

        # 获取右边界（z最大且大于0的点）
        right_point = sorted_points[sorted_points[:, 2] > 0]
        if len(right_point) > 0:
            right_boundaries.append(right_point[-1])  # 右边界是最大的 z 值点

    return left_boundaries, right_boundaries

def detect_and_filter_boundaries(boundaries, z_threshold=0.05, noise_threshold=3.35):
    smoothed_boundaries = []
    prev_z_value = None
    potential_station = False  # 标记是否进入站台区域

    for i, boundary in enumerate(boundaries):
        current_z_value = boundary[2]

        # 如果是站台区域，直接标记为 potential_station
        if prev_z_value is not None:
            z_diff = current_z_value - prev_z_value
            if abs(z_diff) > z_threshold:  # 如果变化超过阈值，认为是异常或站台
                potential_station = True  # 进入站台区域
            else:
                smoothed_boundaries.append(boundary)
                prev_z_value = current_z_value  # 更新上一个 z 值
                continue
        if not potential_station:
            # 噪声判断: 如果 z 值过大，认为是噪声
            if abs(current_z_value) > noise_threshold or abs(current_z_value) < 3:
                continue  # 跳过这个断面
            # 不是站台区域时，保留当前边界
            else:
                smoothed_boundaries.append(boundary)
  
            prev_z_value = current_z_value  # 更新上一个 z 值

    return smoothed_boundaries

# 聚类处理：将点聚成4类
def cluster_points(rail_points, n_clusters=4):
    """
    对钢轨点进行KMeans聚类，强制聚成4类
    n_clusters: 聚类的类别数量
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(rail_points[:, [1, 2]])  # 使用y z坐标进行聚类
    return cluster_labels  # 返回聚类标签

# 计算每个断面的偏心距离
def calculate_eccentricity_per_section(points, labels, cluster_labels, left_boundaries, right_boundaries, z_threshold=0.05):
    eccentricities = []
    rail_left_zs = []  # 存储每个断面左钢轨的 Z 坐标中点
    rail_right_zs = []  # 存储每个断面右钢轨的 Z 坐标中点
    bridge_left_zs = []  # 存储每个断面左人行道边界的 Z 值
    bridge_right_zs = []  # 存储每个断面右人行道边界的 Z 值

    # 获取断面 x 值
    unique_x_values = np.unique(points[:, 0])

    for x in unique_x_values:
        # 获取当前断面的人行道边界点
        left_boundary = [point for point in left_boundaries if point[0] == x]
        right_boundary = [point for point in right_boundaries if point[0] == x]

        # 如果找不到左右边界，跳过当前断面
        if not left_boundary or not right_boundary:
            continue

        left_boundary = left_boundary[0]
        right_boundary = right_boundary[0]

        # 获取当前断面对应的钢轨点
        rail_points = points[labels == 3]  # 假设钢轨标签是3
        rail_points_x = rail_points[rail_points[:, 0] == x]  # 当前断面钢轨点

        # 聚类结果
        cluster_labels_x = cluster_labels[rail_points[:, 0] == x]
        
        # 计算每个聚类的 Z 坐标最小值和最大值，然后计算中点
        cluster_z_centers = []
        for cluster_id in np.unique(cluster_labels_x):
            cluster_points = rail_points_x[cluster_labels_x == cluster_id]
            z_min = np.min(cluster_points[:, 2])  # Z 坐标最小值
            z_max = np.max(cluster_points[:, 2])  # Z 坐标最大值
            z_center = (z_min + z_max) / 2  # Z 坐标中点
            cluster_z_centers.append((cluster_id, z_center))

        # 按 Z 坐标中点排序，最小的为左钢轨，最大的为右钢轨
        cluster_z_centers.sort(key=lambda x: x[1])  # 按 Z 坐标中点从小到大排序

        # 获取左钢轨和右钢轨的聚类
        rail_left_z = cluster_z_centers[0][1]  # Z 值最小的聚类为左钢轨
        rail_right_z = cluster_z_centers[-1][1]  # Z 值最大的聚类为右钢轨

        # 获取对应的钢轨点
        rail_left_points = rail_points_x[cluster_labels_x == cluster_z_centers[0][0]]
        rail_right_points = rail_points_x[cluster_labels_x == cluster_z_centers[-1][0]]

        if rail_left_points.shape[0] == 0 or rail_right_points.shape[0] == 0:
            continue  # 如果没有左钢轨或右钢轨点，跳过此断面

        bridge_left_z = left_boundary[2]
        bridge_right_z = right_boundary[2]

        # 计算偏心距离
        eccentricity = (rail_right_z - bridge_left_z - bridge_right_z + rail_left_z) / 2
        eccentricities.append(eccentricity)

        # 存储每个断面的计算结果
        rail_left_zs.append(rail_left_z)
        rail_right_zs.append(rail_right_z)
        bridge_left_zs.append(bridge_left_z)
        bridge_right_zs.append(bridge_right_z)

    return rail_left_zs, rail_right_zs, bridge_left_zs, bridge_right_zs, eccentricities
def process_folder(folder_path):
    """
    读取文件夹中的所有 .txt 文件并计算偏心距离
    """
    all_eccentricities = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing {file_name}...")
            points, labels = load_and_process_txt(file_path)

            # 获取人行道的左右边界点
            left_boundaries, right_boundaries = get_sidewalk_boundaries(points, labels)

            # 过滤异常值
            filtered_left_boundaries = detect_and_filter_boundaries(left_boundaries, z_threshold=0.05)
            filtered_right_boundaries = detect_and_filter_boundaries(right_boundaries, z_threshold=0.05)

            # 获取钢轨点和聚类结果
            rail_points = points[labels == 3]
            cluster_labels = cluster_points(rail_points, n_clusters=4)

            # 计算每个断面的偏心距离
            rail_left_zs, rail_right_zs, bridge_left_zs, bridge_right_zs, eccentricities = calculate_eccentricity_per_section(
                points, labels, cluster_labels, filtered_left_boundaries, filtered_right_boundaries
            )

            all_eccentricities.extend(eccentricities)

    return all_eccentricities

if __name__ == "__main__":
    folder_path = "E:/lcy/dgcnn/data/coordinate_correction/test_result/dgcnn/seg_1024_rgbrmuda/alpha0.75/K1=50/"  # 文件夹路径
    start_time = time.time()
    eccentricities = process_folder(folder_path)
    print("time: %.4f" % (time.time() - start_time))

    print("\nmean of eccentricities:")
    print(np.mean(eccentricities)*1000)
    print("\nmax of eccentricities:")
    print(np.max(eccentricities)*1000)
    print(np.min(eccentricities)*1000)
