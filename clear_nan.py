import os
input_filename = 'E:/CODE/dgcnn/pytorch/data/coordinate_correction/txt/clean1_split_x_1624_2011_01.txt'
output_filename = 'E:/CODE/dgcnn/pytorch/data/coordinate_correction/txt/cleaned_data.txt'

input_directory = 'E:/lcy/dgcnn/data/coordinate_correction/test1/'
output_directory = 'E:/lcy/dgcnn/data/coordinate_correction/'
# 确保输出目录存在
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 遍历输入目录中的所有文件
for filename in os.listdir(input_directory):
    if filename.endswith('.txt'):  # 确保处理的是TXT文件
        input_filename = os.path.join(input_directory, filename)
        output_filename = os.path.join(output_directory, filename)

        # 读取文件并处理数据
        with open(input_filename, 'r') as file:
            lines = file.readlines()

        # 处理数据，去掉最后两列中的nan，并确保只有9列
        cleaned_data = []
        for line in lines:
            # 分割每一行的数据
            parts = line.strip().split()
            # 确保有且仅有9列
            while len(parts) > 9:
                if parts[-1] == 'nan':
                    parts.pop()  # 去掉最后一列
                elif parts[-2] == 'nan':
                    parts.pop(-2)  # 去掉倒数第二列
                else:
                    break  # 如果没有nan，保持原样

            # 如果不足9列，用0填充
            while len(parts) < 9:
                parts.append('0')

            # 将处理后的数据行添加到结果列表中
            cleaned_data.append(' '.join(parts))

        # 将清洗后的数据写入新文件
        with open(output_filename, 'w') as file:
            for line in cleaned_data:
                file.write(line + '\n')

        print(f"Cleaned data has been written to {output_filename}")