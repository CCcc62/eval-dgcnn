import laspy as las
import numpy as np
import os
from tqdm import tqdm
from matplotlib import cm
import open3d as o3d
# 假设demo模块和scanModel已经正确定义
import demo

def get_deviation_from_log(log_file_path, las_file_path):
    # 读取日志文件并提取偏差值
    with open(log_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    deviations = {}
    las_file_name = las_file_path.split("\\")[-1].split(".")[0]  # 获取.las文件的名称（不含扩展名）
    for line in lines:
        # 查找包含"正在处理文件"的行，以获取日志中对应的.las文件名
        if "正在处理文件" in line:
            parts = line.split(": ")
            if len(parts) > 1:
                # 提取日志中的文件路径，并获取文件名（不含扩展名）
                log_file_name_with_path = parts[-1].strip()
                log_file_name = log_file_name_with_path.split("/")[-1].split(".")[0]
                # 检查日志行中的文件名是否与提供的.las文件名匹配
                if las_file_name == log_file_name:
                    # 找到匹配的文件，现在查找对应的偏差值
                    for subsequent_line in lines[lines.index(line)+1:]:
                        if "坐标偏差值" in subsequent_line:
                            parts = subsequent_line.split("：")
                            if len(parts) > 1:
                                deviation = float(parts[1].strip())
                                deviations[las_file_path] = deviation / 2  # 将偏差值除以2并存储
                            break  # 找到偏差值后跳出循环
    return deviations

def process_point_cloud(las_file_path, output_file_path, deviation):
    railOutside = 0.95
    railInside = 0.55
    sleeperLength = 1.2
    
    lasFile = las.read(las_file_path)
    wholePoints = lasFile.xyz
    # 初始化完整点云为原始颜色
    wholeRed = lasFile.red  # 提取原始红色通道
    wholeGreen = lasFile.green  # 提取原始绿色通道
    wholeBlue = lasFile.blue  # 提取原始蓝色通道
    intensity = lasFile.intensity
    pointId = lasFile.point_source_id
    wholePoints = np.concatenate([wholePoints, intensity.reshape([wholePoints.shape[0], 1]), pointId.reshape([wholePoints.shape[0], 1])], axis=1)
    np.save(os.path.join(os.path.dirname(las_file_path), 'tmp.npy'), wholePoints)

    # 重新加载点云数据
    wholePoints = np.load(os.path.join(os.path.dirname(las_file_path), 'tmp.npy'))
    # railwayPoints = wholePoints[np.where((wholePoints[:, 2] >= -sleeperLength + deviation) & (wholePoints[:, 2] <= sleeperLength + deviation))]
    # railwayPoints = railwayPoints[np.where(railwayPoints[:, 1] <= 0)]
    # railwayPoints = railwayPoints[np.where(railwayPoints[:, 1] >= demo.scanModel[0][1])]
    # 保留 sleeperLength 之内的点
    railwayPoints_within = wholePoints[np.where((wholePoints[:, 2] >= -sleeperLength + deviation) & (wholePoints[:, 2] <= sleeperLength + deviation))]
    # 进行伪彩映射的点，sleeperLength 之外的点
    railwayPoints_outside = wholePoints[np.where((wholePoints[:, 2] < -sleeperLength + deviation) | (wholePoints[:, 2] > sleeperLength + deviation))]
    # 按z值过滤
    #railwayPoints = railwayPoints[np.where((railwayPoints[:, 2] <= -railOutside) | ((-railInside <= railwayPoints[:, 2]) & (railwayPoints[:, 2] <= railInside)) | (railwayPoints[:, 2] >= railOutside))]

    minY = np.min(railwayPoints_outside[:, 1])
    maxY = np.max(railwayPoints_outside[:, 1])

    gray = railwayPoints_outside[:, 1].copy()
    for i in tqdm(range(gray.shape[0]), desc='计算灰度值'):
        if minY <= gray[i] <= maxY:
            gray[i] = 255 * (gray[i] - minY) / (maxY - minY)
        else:
            gray[i] = 0

    # 创建颜色映射
    red = gray.copy()
    green = gray.copy()
    blue = gray.copy()

    cmap = [cm.get_cmap('gist_ncar')(i) for i in range(256)]
    for i in tqdm(range(gray.shape[0]), desc='伪彩映射'):
        color = cmap[int(gray[i])]
        red[i], green[i], blue[i] = color[:3]

    # 初始化完整点云为绿色
    # wholeRed = np.zeros(wholePoints.shape[0], dtype=np.uint16)
    # wholeGreen = np.full(wholePoints.shape[0], 255, dtype=np.uint16)
    # wholeBlue = np.zeros(wholePoints.shape[0], dtype=np.uint16)
    

    # 将伪彩映射应用到完整点云的道床部分
    for i, point in tqdm(enumerate(railwayPoints_outside), desc='应用伪彩映射'):
        match_indices = np.where((wholePoints[:, 0] == point[0]) & (wholePoints[:, 1] == point[1]) & (wholePoints[:, 2] == point[2]))[0]
        for idx in match_indices:
            wholeRed[idx] = int(red[i] * 255)
            wholeGreen[idx] = int(green[i] * 255)
            wholeBlue[idx] = int(blue[i] * 255)

    # 创建新的LAS文件头
    header = las.LasHeader(point_format=3, version='1.2')
    header.offsets = np.zeros(3)
    header.scales = np.ones(3) * 1e-4
    newLas = las.LasData(header)
    newLas.x = wholePoints[:, 0]
    newLas.y = wholePoints[:, 1]
    newLas.z = wholePoints[:, 2]
    newLas.red = wholeRed
    newLas.green = wholeGreen
    newLas.blue = wholeBlue
    newLas.intensity = wholePoints[:, 3]
    newLas.point_source_id = wholePoints[:, 4]
    newLas.write(output_file_path)

# 批量处理点云文件
def batch_process_point_clouds(input_directory, output_directory, log_file_path):
    for filename in os.listdir(input_directory):
        if filename.endswith('.las'):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            deviation = get_deviation_from_log(log_file_path, filename).get(filename, 0)  # 获取偏差值
            process_point_cloud(input_path, output_path, deviation)
            print(f"Processed {input_path} -> {output_path}")

def process_point_cloud_txt(txt_file_path, output_file_path, deviation):
    railOutside = 0.95
    railInside = 0.55
    sleeperLength = 1.2
    # 读取 TXT 文件
    data = np.loadtxt(txt_file_path)
    points = data[:, :3]    # XYZ
    colors = data[:, 3:6]   # RGB
    labels = data[:, -1]     # 标签
    additional_info = data[:, 6:-1]  # RGB 和标签之间的所有其他信息
    # 将标签重塑为二维数组
    labels = labels.reshape([-1, 1])
    wholePoints = np.concatenate([points, colors, additional_info, labels], axis=1)
    # 初始化点云颜色为绿色 最后对sleeperLength 之外的点伪彩映射时取消这一步
    # green_color = np.array([0, 255, 0], dtype=np.uint8)
    # wholePoints[:, 3:6] = green_color
    # 保留 sleeperLength 之内的点
    railwayPoints_within = wholePoints[np.where((wholePoints[:, 2] >= -sleeperLength + deviation) & (wholePoints[:, 2] <= sleeperLength + deviation))]
    # 进行伪彩映射的点，sleeperLength 之外的点
    railwayPoints_outside = wholePoints[np.where((wholePoints[:, 2] < -sleeperLength + deviation) | (wholePoints[:, 2] > sleeperLength + deviation))]

    # 计算伪彩映射的灰度值
    minY = np.min(railwayPoints_outside[:, 1])
    maxY = np.max(railwayPoints_outside[:, 1])

    gray = railwayPoints_outside[:, 1].copy()
    for i in tqdm(range(gray.shape[0]), desc='计算灰度值'):
        if minY <= gray[i] <= maxY:
            gray[i] = 255 * (gray[i] - minY) / (maxY - minY)
        else:
            gray[i] = 0

    # 创建颜色映射
    red = gray.copy()
    green = gray.copy()
    blue = gray.copy()

    cmap = [cm.get_cmap('gist_ncar')(i) for i in range(256)]
    for i in tqdm(range(gray.shape[0]), desc='伪彩映射'):
        color = cmap[int(gray[i])]
        red[i], green[i], blue[i] = color[:3]

    # 更新伪彩映射应用到完整点云的 sleeperLength 之外的部分
    for i, point in tqdm(enumerate(railwayPoints_outside), desc='应用伪彩映射'):
        match_indices = np.where((wholePoints[:, 0] == point[0]) & 
                                 (wholePoints[:, 1] == point[1]) & 
                                 (wholePoints[:, 2] == point[2]))[0]
        for idx in match_indices:
            wholePoints[idx, 3] = int(red[i] * 255)  # 更新红色通道
            wholePoints[idx, 4] = int(green[i] * 255)  # 更新绿色通道
            wholePoints[idx, 5] = int(blue[i] * 255)  # 更新蓝色通道

    # 保存处理好的结果为 TXT 文件
    fmt = '%.8f %.8f %.8f %d %d %d ' + ' '.join(['%.6f'] * additional_info.shape[1]) + ' %.6f'
    np.savetxt(output_file_path, wholePoints, fmt=fmt)

# 批量处理点云文件
def batch_process_point_clouds_txt(input_directory, output_directory, log_file_path):
    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)  # 保存为同名文件
            deviation = get_deviation_from_log(log_file_path, filename).get(filename, 0)  # 获取偏差值
            process_point_cloud_txt(input_path, output_path, deviation)
            print(f"Processed {input_path} -> {output_path}")

# 设置输入和输出目录
input_directory = "E:/lcy/dgcnn/data/coordinate_correction/bridge2_proceed/"
output_directory = "E:/lcy/dgcnn/data/coordinate_correction/bridge2_proceed_las/"
log_file_path = "E:/lcy/dgcnn/output/2024_12_31_11_28/log.txt"  # 日志文件路径
batch_process_point_clouds(input_directory, output_directory, log_file_path)
#batch_process_point_clouds_txt(input_directory, output_directory, log_file_path)
