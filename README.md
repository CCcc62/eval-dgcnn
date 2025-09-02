本仓库是基于DGCNN pytorch(https://github.com/antao97/dgcnn.pytorch)实现版本实现的。

项目概述：
本项目是基于DGCNN PyTorch实现的铁路桥梁偏心距离测量系统，主要贡献如下：
1）针对目前国内领域空白的桥梁偏心距离自动化测量任务，首个提出精确的对应测量方法，解决传统方法效率低、不方便且存在人工误差以及引进国外设备价格昂贵的问题。
2）为了解决原始分割网络出现的语义混淆问题，根据铁路桥梁特有的结构先验约束，本文从网络输入以及网络结构两个方面对原始分割网络进行改进，使其更适合本文的数据，提高分割精度。
3）即使改进之后的网络分割精度提高，但铁路桥梁偏心距离精度仍然较低，本文利用其他桥梁结构先验约束提出了合适的标签纠正算法，提高了铁路桥梁偏心计算精度。
主要功能：
1.点云数据预处理
项目包含LAS文件分割、处理，数据清洗，去噪等，具体对应的代码文件见项目结构说明

2.点云语义分割
本仓库使用ElevationWeightedDGCNN模型对点云数据进行分割，输入为h5文件，输出为txt文件。如有需要，也可以选择pointnet等模型，详细请看model.py

3.桥梁偏心距离计算
桥梁偏心距离为轨道中心与桥梁整体中心的偏差值，eccentricDistanceCal.py 提供了完整的偏心距离计算功能

项目结构：
├── main_seg.py              # 主要的分割训练和测试脚本，包含标签矫正实现
├── model.py                 # 神经网络模型定义
├── data.py                  # 数据加载
├── util.py                  # 工具函数
├── eccentricDistanceCal.py  # 偏心距离计算
├── createLasfile.py         # LAS文件处理，根据高程差编码为颜色信息
├── cutLasfile.py           # LAS文件切分
├── txttoh5.py              # 数据格式转换，txt文件转换为h5文件，模型训练测试使用h5格式文件
├── clear_nan.py            # 数据清洗，去除数据中的异常值
├── statisticalFiltering.py  # 统计滤波去噪
├── demo.py                 # 演示程序
└── countPoints.py          # 点云统计

Requirements:
Python >= 3.7
PyTorch >= 1.2
CUDA >= 10.0
package:
pip install torch torchvision
pip install numpy scipy scikit-learn
pip install h5py glob2 plyfile
pip install laspy open3d matplotlib
pip install tqdm loguru opencv-python

快速开始
1.数据准备
数据分割，将整段点云数据分割为合适的训练数据
python cutLasfile.py
去噪
python statisticalFiltering.py
根据高程差编码为rgb，与xyz一起作为输入
python createLasfile.py
数据清洗
python clear_nan.py
转换LAS文件为训练格式，训练以及测试数据放在data文件夹下，或者自己设置数据路径
python txttoh5.py

2.模型训练，参数可以在代码中设置或者在指令中指定，训练结果保存在outputs文件夹中
python main_seg.py

3.模型测试
python main_seg.py --eval==True

4.偏心距离计算
python eccentricDistanceCal.py

联系方式
如有任何文件或建议，请通过Issue或者邮件联系
