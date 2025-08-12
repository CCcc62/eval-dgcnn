import laspy
import numpy as np
import time
def slice_and_filter_by_z_density(input_las_path, output_las_path, x_slice_size, z_bin_size, density_threshold, z_ranges):
    # 读取 LAS 文件
    las = laspy.read(input_las_path)
    start_time = time.time()
    # 获取点云数据
    x_coords = las.x
    z_coords = las.z

    # 计算 x 方向切片的范围
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    x_bins = np.arange(x_min, x_max + x_slice_size, x_slice_size)

    # 初始化有效的索引
    valid_indices = np.arange(len(las.points))  # 先将所有点的索引视为有效

    # 第一步：遍历每个 x 方向的切片
    for i in range(len(x_bins) - 1):
        x_start = x_bins[i]
        x_end = x_bins[i + 1]

        # 筛选当前切片中的点
        slice_indices = np.where((x_coords >= x_start) & (x_coords < x_end))[0]
        slice_z_coords = z_coords[slice_indices]

        # 第二步：统计指定 z 方向范围内的点云密度并筛选噪声
        for z_range in z_ranges:
            range_indices = np.where((slice_z_coords >= z_range[0]) & (slice_z_coords <= z_range[1]))[0]
            if len(range_indices) > 0:
                z_min, z_max = z_range
                z_bins = np.arange(z_min, z_max + z_bin_size, z_bin_size)
                density, _ = np.histogram(slice_z_coords[range_indices], z_bins)

                # 第三步：遍历密度筛选噪声
                for j in range(len(density)):
                    if density[j] < density_threshold:  # 如果密度小于阈值
                        # 筛选当前 z 范围内的索引
                        invalid_indices = slice_indices[(slice_z_coords >= z_bins[j]) & (slice_z_coords < z_bins[j + 1])]
                        valid_indices = np.setdiff1d(valid_indices, invalid_indices)  # 从有效索引中去除这些噪声点
    print("time:%.4f", time.time() - start_time)
    # 保存去除噪声后的完整点云
    new_las = laspy.LasData(las.header)
    new_las.points = las.points[valid_indices]

    # 将新的 LAS 文件写入磁盘
    new_las.write(output_las_path)

def slice_and_filter_by_y_density(input_las_path, output_las_path, x_slice_size=1, y_bin_size=0.01, density_threshold=50, z_ranges=[(-3.7, -3.4), (3.4, 3.7)]):
    # 读取 LAS 文件
    las = laspy.read(input_las_path)
    start_time = time.time()
    # 获取点云数据
    x_coords = las.x
    y_coords = las.y
    z_coords = las.z

    # 计算 x 方向切片的范围
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    x_bins = np.arange(x_min, x_max + x_slice_size, x_slice_size)

    # 初始化有效的索引
    valid_indices = np.arange(len(las.points))  # 先将所有点的索引视为有效

    # 第一步：遍历每个 x 方向的切片
    for i in range(len(x_bins) - 1):
        x_start = x_bins[i]
        x_end = x_bins[i + 1]

        # 筛选当前切片中的点
        slice_indices = np.where((x_coords >= x_start) & (x_coords < x_end))[0]
        slice_y_coords = y_coords[slice_indices]
        slice_z_coords = z_coords[slice_indices]

        # 第二步：遍历指定的 z 范围
        for z_range in z_ranges:
            range_indices = slice_indices[(slice_z_coords >= z_range[0]) & (slice_z_coords < z_range[1])]
            range_y_coords = y_coords[range_indices]

            # 第三步：统计 y 方向的点云密度
            if len(range_y_coords) > 0:
                y_min, y_max = np.min(range_y_coords), np.max(range_y_coords)
                y_bins = np.arange(y_min, y_max + y_bin_size, y_bin_size)
                density, _ = np.histogram(range_y_coords, y_bins)

                # 第四步：遍历密度筛选噪声
                for j in range(len(density)):
                    if density[j] < density_threshold:  # 如果密度小于阈值
                        # 筛选当前 y 范围内的索引
                        invalid_indices = range_indices[(range_y_coords >= y_bins[j]) & (range_y_coords < y_bins[j + 1])]
                        valid_indices = np.setdiff1d(valid_indices, invalid_indices)  # 从有效索引中去除这些噪声点
    print("time:%.4f", time.time() - start_time)
    # 保存去除噪声后的完整点云
    new_las = laspy.LasData(las.header)
    new_las.points = las.points[valid_indices]

    # 将新的 LAS 文件写入磁盘
    new_las.write(output_las_path)

def filter_by_y_value(input_las_path, output_las_path, z_range1=(-3.7, -3.3), z_range2=(3.3, 3.7), y_threshold1=0.5, y_threshold2=0.5):
    # 读取 LAS 文件
    las = laspy.read(input_las_path)
    start_time = time.time()
    # 获取点云数据
    y_coords = las.y
    z_coords = las.z

    # 初始化有效的索引
    valid_indices = np.arange(len(las.points))  # 先将所有点的索引视为有效

    # 定义要筛选的z范围
    z_ranges = [z_range1, z_range2]
    y_thresholds = [y_threshold1, y_threshold2]

    # 遍历每个z范围并进行y值筛选
    for z_range, y_threshold in zip(z_ranges, y_thresholds):
        # 筛选当前z范围内的点
        range_indices = np.where((z_coords > z_range[0]) & (z_coords < z_range[1]))[0]
        range_y_coords = y_coords[range_indices]

        # 遍历y值筛选噪声
        invalid_indices = range_indices[range_y_coords < y_threshold]
        valid_indices = np.setdiff1d(valid_indices, invalid_indices)  # 从有效索引中去除这些噪声点
    print("time:%.4f", time.time() - start_time)
    # 保存去除噪声后的完整点云
    new_las = laspy.LasData(las.header)
    new_las.points = las.points[valid_indices]

    # 将新的 LAS 文件写入磁盘
    new_las.write(output_las_path)

# 示例调用
input_las_path = 'E:/lcy/项目数据/大理/test2.las'
output_las_path = 'E:/lcy/项目数据/大理/test3.las'
# slice_and_filter_by_z_density(input_las_path, output_las_path, x_slice_size=1, z_bin_size=0.5, density_threshold=150, z_ranges=[[-3.7, -3.2], [3.2, 3.7]])
# filter_by_y_value(input_las_path, output_las_path, z_range1=(-3.7, -3.3), z_range2=(3.3, 3.7), y_threshold1=-0.95, y_threshold2=-0.65)
slice_and_filter_by_y_density(input_las_path, output_las_path, x_slice_size=1, y_bin_size=0.01, density_threshold=10, z_ranges=[(-3.7, -3.5), (3.5, 3.7)])