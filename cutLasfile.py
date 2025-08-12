import laspy
import numpy as np
import os

def split_las_by_x_by_chunk_size(input_file, output_dir, x_intervals, chunk_size=1000):
    """
    按照x坐标分割 LAS 文件
    :param input_file: 输入 LAS 文件路径
    :param output_dir: 输出目录
    :param x_intervals: x坐标区间列表，例如 [(0, 10), (10, 20)]
    :param chunk_size: 每次读取的点数
    """
    # 读取原始 LAS 文件
    las = laspy.read(input_file)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 按x坐标区间创建并保存新的 LAS 文件
    total_points = las.header.point_count
    for start, end in x_intervals:
        selected_points = []
        # 分块读取点云数据
        for start_idx in range(0, total_points, chunk_size):
            end_idx = min(start_idx + chunk_size, total_points)
            chunk = las[start_idx:end_idx]
            # 根据x坐标筛选点云
            mask = (chunk.x >= start) & (chunk.x < end)
            selected_points.extend(chunk[mask])
        
        # 创建新的 LAS 文件
        new_las = laspy.LasData(las.header)
        new_las.points = selected_points

        # 保存新的 LAS 文件到指定目录
        output_file_path = os.path.join(output_dir, f"split_x_{start}_{end}.las")
        new_las.write(output_file_path)
        print(f"Saved split LAS file: {output_file_path}")

def split_las_by_x(input_file, output_dir, x_intervals):
    # 读取原始 LAS 文件
    las = laspy.read(input_file)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 按x坐标区间创建并保存新的 LAS 文件
    for start, end in x_intervals:
        # 根据x坐标筛选点云
        mask = (las.x >= start) & (las.x <= end)
        selected_points = las.points[mask]
        
        # 创建新的 LAS 文件
        new_las = laspy.LasData(las.header)
        
        # 将筛选的点赋值给新的LasData对象的points属性
        new_las.points = selected_points

        # 保存新的 LAS 文件到指定目录
        output_file_path = os.path.join(output_dir, f"split_x_{start}_{end}.las")
        new_las.write(output_file_path)
        print(f"Saved split LAS file: {output_file_path}")

def split_las_by_z(input_file, output_dir, z_intervals):
    # 读取原始 LAS 文件
    las = laspy.read(input_file)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 按x坐标区间创建并保存新的 LAS 文件
    for start, end in z_intervals:
        # 根据x坐标筛选点云
        mask = (las.z >= start) & (las.z <= end)
        selected_points = las.points[mask]
        
        # 创建新的 LAS 文件
        new_las = laspy.LasData(las.header)
        
        # 将筛选的点赋值给新的LasData对象的points属性
        new_las.points = selected_points

        # 保存新的 LAS 文件到指定目录
        output_file_path = os.path.join(output_dir, f"split_z_{start}_{end}.las")
        new_las.write(output_file_path)
        print(f"Saved split LAS file: {output_file_path}")

def split_las_by_y(input_file, output_dir, y_intervals):
    # 读取原始 LAS 文件
    las = laspy.read(input_file)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 按x坐标区间创建并保存新的 LAS 文件
    for start, end in y_intervals:
        # 根据x坐标筛选点云
        mask = (las.y >= start) & (las.y <= end)
        selected_points = las.points[mask]
        
        # 创建新的 LAS 文件
        new_las = laspy.LasData(las.header)
        
        # 将筛选的点赋值给新的LasData对象的points属性
        new_las.points = selected_points

        # 保存新的 LAS 文件到指定目录
        output_file_path = os.path.join(output_dir, f"split_y_{start}_{end}.las")
        new_las.write(output_file_path)
        print(f"Saved split LAS file: {output_file_path}")

def split_las_into_equal_parts(input_file, output_dir, num_parts):
    """
    将输入的 LAS 文件平均分成指定数量的部分
    :param input_file: 输入 LAS 文件路径
    :param output_dir: 输出目录
    :param num_parts: 要分割的部分数量
    """
    # 读取原始 LAS 文件
    las = laspy.read(input_file)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_points = las.header.point_count
    part_size = total_points // num_parts

    for i in range(num_parts):
        start_idx = i * part_size
        if i == num_parts - 1:  # 确保最后一个部分包含所有剩余的点
            end_idx = total_points
        else:
            end_idx = (i + 1) * part_size

        selected_points = las.points[start_idx:end_idx]
        
        # 创建新的 LAS 文件
        new_las = laspy.LasData(las.header)
        new_las.points = selected_points

        # 保存新的 LAS 文件到指定目录
        output_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_{i + 1}.las")
        new_las.write(output_file_path)
        print(f"Saved split LAS file: {output_file_path}")

# 示例使用
input_file = "E:/lcy/项目数据/大理/clean1_split_x_3331_3800.las"  # 输入 LAS 文件路径
output_dir = "E:/lcy/项目数据/大理/"  # 输出目录

# 定义里程值区间
mileage_intervals_x = [(1584, 2051)]  # 这里可以根据你的实际需求定义
mileage_intervals_z = [(-3.7, 3.7)]
mileage_intervals_y = [(-1, 1)]
# 分割 LAS 文件
#split_las_by_z(input_file, output_dir, mileage_intervals_z)

# 将点云平均分成num_parts等份
split_las_into_equal_parts(input_file, output_dir, 60)
