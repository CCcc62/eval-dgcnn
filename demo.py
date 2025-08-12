"""
demo程序，展示各种方法。
"""
import copy
import os
import time
from datetime import datetime
from tqdm import tqdm

import cv2
import laspy as las
import numpy as np
import open3d as o3d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib import font_manager
from loguru import logger

#from mmdet3d.apis import LidarSeg3DInferencer
#from mmseg.apis import inference_model, init_model, show_result_pyplot

import repc

def costTime(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        ret = func(*args, **kwargs)
        endTime = time.time()
        if func.__name__ in ['method1d', 'method2d', 'seg2d', 'seg3d', 'windowSlide', 'statisticalSeg', 'errorSeg']:
            totalCostTime = endTime - startTime - ret
        else:
            totalCostTime = endTime - startTime
        logger.info(f"{func.__name__}()耗时：{totalCostTime}s")
        return ret

    return wrapper


def findpeaks(data, spacing=1, limit=None):
    """Finds peaks in `data` which are of `spacing` width and >=`limit`.
    https://zhuanlan.zhihu.com/p/551306898 的 Janko Slavic 的实现
    :param data: values
    :param spacing: minimum spacing to the next peak (should be 1 or more)
    :param limit: peaks should have value greater or equal
    :return:
    """
    len = data.size
    x = np.zeros(len + 2 * spacing)
    x[:spacing] = data[0] - 1.e-6
    x[-spacing:] = data[-1] - 1.e-6
    x[spacing:spacing + len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start: start + len]  # before
        start = spacing
        h_c = x[start: start + len]  # central
        start = spacing + s + 1
        h_a = x[start: start + len]  # after
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind


def errorSeg2(points, show=False):
    slopThr = 8  # 坡度
    heightThr0 = 0.15  # 低于这个值的高度差直接算作噪声
    heightThr1 = 0.08  # 高于这个值的高度差才考虑是否是噪声
    heightThr2 = 0.15   # 平均差值高度高于这个值的平台视为噪声
    pixelWidth = 4e-3 * 3

    # 计算里程信息
    assert points.shape[0] > 0
    mileageList = [[]]
    mileageIndex = 0
    for pointIndex in range(points.shape[0]):
        mileageList[mileageIndex].append(pointIndex)
        if pointIndex < points.shape[0] - 1:
            if points[pointIndex + 1, 0] != points[pointIndex, 0]:
                mileageIndex += 1
                mileageList.append([])

    mileageNum = len(mileageList)
    mileageMeanInterval = abs(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (
            mileageNum - 1)
    if mileageNum > 1:
        logger.info(
            f"里程的排列是否均匀："
            f"{(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (points[mileageList[0][0], 0] - points[mileageList[1][0], 0]) == mileageNum - 1}，"
            f"平均断面间隔为：{mileageMeanInterval}m")
    else:
        logger.error("里程数不足！")

    pointColors = np.zeros([points.shape[0], 3])
    pointColors[:] = [0, 255, 0]  # 先为所有点附上绿色

    for mileageIndex, sectionList in tqdm(enumerate(mileageList)):
        errorList = []
        borderIndex = 0
        leftDescend = 0
        sectionIndex = 1
        heightMean = 0  # 除去噪声点后的平均高度
        heightMeanNum = 0
        tmpHeight = 0   # 临时计算的平均高度
        tmpNum = 0
        while sectionIndex < len(sectionList):
            tmpNum += 1
            tmpHeight += points[sectionList[sectionIndex], 1] - modelY(points[sectionList[sectionIndex], 2])
            # 计算当前差分值
            rightDescend = points[sectionList[sectionIndex], 1] - points[sectionList[sectionIndex - 1], 1]
            # 当前差分与之前差分乘积为负数，即拐点
            if leftDescend * rightDescend < 0:
                # 计算两个拐点间的高度差
                height = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[borderIndex], 1])
                # 高度差必要条件
                if height >= heightThr1:
                    width = points[sectionList[sectionIndex - 1], 2] - points[sectionList[borderIndex], 2]
                    if width == 0:
                        # 则该点必是噪声
                        errorList.append(sectionList[borderIndex])
                        errorList.append(sectionList[sectionIndex - 1])
                        # 噪声则临时均值归零
                        tmpHeight = 0
                        tmpNum = 0
                    else:
                        slop = height / width
                        # 大于高度差充分条件或者坡度在阈值内的才计入轨枕
                        if height >= heightThr0 or slop >= slopThr:
                            errorList.append(sectionList[borderIndex])
                            errorList.append(sectionList[sectionIndex - 1])
                            # 噪声则临时均值归零
                            tmpHeight = 0
                            tmpNum = 0
                borderIndex = sectionIndex - 1
                leftDescend = rightDescend
                # 每逢拐点则加一次均值
                heightMean += tmpHeight
                heightMeanNum += tmpNum
            elif leftDescend == 0:
                leftDescend = rightDescend
            sectionIndex += 1

        # 计算均值
        heightMean /= heightMeanNum
        # 筛选边界点与边界点之间
        finalList = []
        errorList.insert(0, sectionList[0])
        errorList.append(sectionList[-1])
        errorIndex = 0
        while errorIndex < len(errorList):
            leftIndex = errorList[errorIndex]
            rightIndex = errorList[errorIndex + 1]
            meanDiff = 0
            for pIndex in range(leftIndex, rightIndex):
                meanDiff += abs(points[pIndex, 1] - modelY(points[pIndex, 2]) - heightMean)
            if rightIndex - leftIndex > 0:
                meanDiff /= (rightIndex - leftIndex)
            if meanDiff >= heightThr2:
                finalList.append(leftIndex)
                if errorIndex < len(errorList) - 2:
                    finalList.append(errorList[errorIndex + 2])
                else:
                    finalList.append(rightIndex)
            errorIndex += 2

        # 标注点云
        pointIndex = 0
        while pointIndex < len(finalList):
            leftIndex = finalList[pointIndex]
            rightIndex = finalList[pointIndex + 1]
            for pIndex in range(leftIndex, rightIndex + 1):
                pointColors[pIndex] = [255, 0, 0]  # 轨枕部分用红色标注
            pointIndex += 2

    if show:
        showPointCloud(points, pointColors)


def railSeg(points, show=False):
    """
    基于一维统计信号的分割方法。
    :param points: (np.array[n, 4]) 输入的点云
    :param show: (bool) 是否展示结果
    :return: None
    """
    sectionInterval = 5e-3
    showCostTime = 0.0

    railOutside = 0.95
    railInside = 0.55

    # 计算里程信息
    assert points.shape[0] > 0
    mileageList = [[]]
    mileageIndex = 0
    for pointIndex in range(points.shape[0]):
        mileageList[mileageIndex].append(pointIndex)
        if pointIndex < points.shape[0] - 1:
            if points[pointIndex + 1, 0] != points[pointIndex, 0]:
                mileageIndex += 1
                mileageList.append([])

    mileageNum = len(mileageList)
    mileageMeanInterval = abs(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (
            mileageNum - 1)
    '''
    if mileageNum > 1:
        logger.info(
            f"里程的排列是否均匀：{(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (points[mileageList[0][0], 0] - points[mileageList[1][0], 0]) == mileageNum - 1}，"
            f"平均断面间隔为：{mileageMeanInterval}m")
    else:
        logger.error("里程数不足！")
        return showCostTime'''

    # 生成差分图
    _, diffMesh, mesh2points, _, _, _, _ = generateDiffImage(
        points, mileageList, mileageNum, mileageMeanInterval, scanModel[2][0], scanModel[3][0],
        5e-3, useInterpolation=False)

    # 使用示例
    timeStr = datetime.utcnow().strftime('%H%M%S_%f')[:-3]  # 获取当前时间字符串
    save_diffmesh_image(diffMesh, timeStr, outputDir)
    # 抹平钢轨屏蔽区
    # rail1 = int((-railOutside - scanModel[2][0]) / 5e-3)
    # rail2 = int((-railInside - scanModel[2][0]) / 5e-3)
    # rail3 = int((railInside - scanModel[2][0]) / 5e-3)
    # rail4 = int((railOutside - scanModel[2][0]) / 5e-3)

    # diffMesh[:, rail1: rail2] = 0
    # diffMesh[:, rail3: rail4] = 0

    colMeanInfo = np.zeros((diffMesh.shape[1],))
    # 逐列像素求均值和标准差
    for col in range(diffMesh.shape[1]):
        colMeanInfo[col] = np.mean(diffMesh[:, col])

    # 铁轨按列排布，且根据模型图，位于道砟模型图上方；并且，由于其方差小，所以均值较之于别的像素列要高
    # 由于铁轨有两根，所以只需要找出列均值最大的两个波峰，就算找到了铁轨
    # height参数限制了波峰的下限，此波峰至少要大于整体均值
    peaks = findpeaks(colMeanInfo, spacing=int(0.3 / sectionInterval), limit=0.20)
    # peaks = AMPD(colMeanInfo, height=0.245)
    # peaks, _ = find_peaks(colMeanInfo, height=0.245)

    plt.figure(1)
    plt.clf()
    xTicks = np.arange(scanModel[2][0], scanModel[3][0] + sectionInterval, sectionInterval)
    plt.plot(xTicks, colMeanInfo)
    plt.plot(xTicks, np.ones_like(colMeanInfo) * np.mean(colMeanInfo), '--')
    # 轨道的范围
    railRange = []
    for peak in peaks:
        leftBase, rightBase = findPeakBase(colMeanInfo, peak)
        if leftBase == peak or rightBase == peak:
            continue

        railRange.append([leftBase, rightBase])

        plt.plot(peak * sectionInterval + scanModel[2][0], colMeanInfo[peak], "x")
        plt.plot(leftBase * sectionInterval + scanModel[2][0], colMeanInfo[leftBase], "o")
        plt.plot(rightBase * sectionInterval + scanModel[2][0], colMeanInfo[rightBase], "o")

    plt.xlabel('Point cloud cross-sectional coordinates(m)', fontproperties=myFont)
    plt.ylabel('The mean elevation of the point cloud cross-sectional column(m)', fontproperties=myFont)
    timeStr = datetime.utcnow().strftime('%H%M%S_%f')[:-3]
    # 先设定计算结果保存目录
    method1dDir = os.path.join(outputDir, "method1d")
    os.makedirs(method1dDir, exist_ok=True)
    plt.savefig(os.path.join(method1dDir, f"col_mean_{timeStr}.svg"), dpi=300, format="svg")
    
    if show:
        startTime = time.time()
        plt.show()
        endTime = time.time()
        showCostTime += endTime - startTime
    railLeftZ, leftCount, railRightZ, rightCount = 0, 0, 0, 0
    guardRailLeftZ, guardRailLeftCount, guardRailRightZ, guardRailRightCount = 0, 0, 0, 0
    # 根据找到的范围，提取出分割后的点云
    pointColors = np.zeros([points.shape[0], 3])
    pointColors[:] = [0, 255, 0]  # 先为所有点附上绿色
    for rail in railRange:
        for colIndex in range(rail[0], rail[1]):
            for rowIndex in range(diffMesh.shape[0]):
                for pIndex in mesh2points[rowIndex][colIndex]:
                    pointColors[pIndex] = [0, 0, 255]  # 钢轨部分用蓝色标注
    for rowIndex in range(diffMesh.shape[0]):
        for pIndex in mesh2points[rowIndex][railRange[0][1]]:
            railLeftZ += points[pIndex][2]
            leftCount += 1
        for pIndex in mesh2points[rowIndex][railRange[-1][0]]:
            railRightZ += points[pIndex][2]
            rightCount += 1
    railLeftZ /= leftCount
    railRightZ /= rightCount
    logger.info(
        f"坐标偏差值：{railRightZ - abs(railLeftZ)}")       
    if show:
        startTime = time.time()
        showPointCloud(points, pointColors)
        endTime = time.time()
        showCostTime += endTime - startTime
    
    # 如果检测出4根钢轨
    if len(peaks) == 4:
        for rowIndex in range(diffMesh.shape[0]):
            for pIndex in mesh2points[rowIndex][railRange[1][1]]:
                guardRailLeftZ += points[pIndex][2]
                guardRailLeftCount += 1
            for pIndex in mesh2points[rowIndex][railRange[2][0]]:
                guardRailRightZ += points[pIndex][2]
                guardRailRightCount += 1
        guardRailLeftZ /= guardRailLeftCount
        guardRailRightZ /= guardRailRightCount
        # 如果和主轨保持一定的距离并且距离相等
        if abs(railLeftZ -  guardRailLeftZ) > 0.5 and abs(railRightZ -  guardRailRightZ) > 0.5 and abs(abs(railLeftZ -  guardRailLeftZ) - abs(railRightZ -  guardRailRightZ)) <= 0.046: 
            bridgeMileageList[-1].append(points[0][0])
            bridgeMileageList[-1].append(points[-1][0])
            
  
def errorSeg(points, show=False):
    """
    用断面进行突变去噪。
    :param points: (np.array) 需要分割的点云
    :param show: (bool) 是否显示结果
    :return: None
    """
    rowDir = os.path.join(outputDir, "rowScanSleeper")
    os.makedirs(rowDir, exist_ok=True)
    showCostTime = 0.0

    yDiffThr = 0.05
    gradientThr = 5
    railOutside = 0.95
    railInside = 0.55

    # 计算里程信息
    assert points.shape[0] > 0
    mileageList = [[]]
    mileageIndex = 0
    for pointIndex in range(points.shape[0]):
        mileageList[mileageIndex].append(pointIndex)
        if pointIndex < points.shape[0] - 1:
            if points[pointIndex + 1, 0] != points[pointIndex, 0]:
                mileageIndex += 1
                mileageList.append([])

    mileageNum = len(mileageList)
    mileageMeanInterval = abs(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (
            mileageNum - 1)
    if mileageNum > 1:
        logger.info(
            f"里程的排列是否均匀："
            f"{(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (points[mileageList[0][0], 0] - points[mileageList[1][0], 0]) == mileageNum - 1}，"
            f"平均断面间隔为：{mileageMeanInterval}m")
    else:
        logger.error("里程数不足！")
        return showCostTime

    for mileageIndex, sectionList in tqdm(enumerate(mileageList)):
        showPlt = False

        sectionNum = len(sectionList)
        sectionArray = np.zeros(sectionNum)
        gradientArray = np.zeros(sectionNum)
        gradientIndexList = []
        zArray = np.zeros(sectionNum)

        lastY = points[sectionList[0], 1]
        lastZ = points[sectionList[0], 2]
        sectionArray[0] = lastY
        zArray[0] = lastZ
        for sectionIndex in range(1, sectionNum):
            nowY = points[sectionList[sectionIndex], 1]
            nowZ = points[sectionList[sectionIndex], 2]
            sectionArray[sectionIndex] = nowY
            zArray[sectionIndex] = nowZ
            if -railOutside <= nowZ <= -railInside or railInside <= nowZ <= railOutside:
                pass
            elif abs(nowY - lastY) < yDiffThr:
                pass
            elif nowZ == lastZ:
                gradientArray[sectionIndex] = (nowY - lastY) * 1e3
                gradientIndexList.append(sectionIndex)
                showPlt = True
            else:
                gradient = (nowY - lastY) / (nowZ - lastZ)
                if abs(gradient) > gradientThr:
                    gradientArray[sectionIndex] = gradient
                    gradientIndexList.append(sectionIndex)
                    showPlt = True
            lastY = nowY
            lastZ = nowZ

        if show:
            plt.clf()
            plt.subplot(3, 1, 1)
            plt.plot(zArray, sectionArray)
            plt.axis("equal")
            plt.grid('on')

            plt.subplot(3, 1, 2)
            plt.plot(zArray, gradientArray)
            plt.grid('on')

        if show and mileageIndex >= 77:
            print('debug')

        sectionMean = float(sectionArray.mean())    # 求均值，待会儿有用
        fixedSectionArray = sectionArray.copy()
        gradientIndex = 0
        while gradientIndex < len(gradientIndexList):
            nowIndex = gradientIndexList[gradientIndex]
            lastIndex = 0
            nextIndex = sectionNum - 1
            if gradientIndex > 0:
                lastIndex = gradientIndexList[gradientIndex - 1]
            if gradientIndex < len(gradientIndexList) - 1:
                nextIndex = gradientIndexList[gradientIndex + 1]
            # # 判断是否为垂直突变
            # if sectionArray[nowIndex] == sectionArray[nowIndex - 1]:
            # 看下一个点与上一个间高程差是否超出阈值
            if nowIndex == sectionNum - 1 or abs(sectionArray[nowIndex - 1] - sectionArray[nowIndex + 1]) < yDiffThr:
                # 未超出阈值，则可能有两种情况：1.该点是异常点，两边都正常；2.该点是正常点，两边都异常！
                if abs(modelY(zArray[nowIndex - 1]) - sectionArray[nowIndex - 1]) > abs(modelY(zArray[nowIndex]) - sectionArray[nowIndex]):
                    # 若中间点离模板更近，则认为这是正常点，则先修正左边
                    fixIndex = nowIndex - 1
                    # 到上一个点为止全是噪声！
                    while fixIndex >= lastIndex and (sectionArray[fixIndex] - sectionMean) >= yDiffThr:
                        fixedSectionArray[fixIndex] = modelY(zArray[fixIndex])

                        # 修改点云值
                        points[sectionList[fixIndex], 1] = fixedSectionArray[fixIndex]

                        fixIndex -= 1
                    # 接着修改右边
                    fixIndex = nowIndex
                    # 到下一个点为止全是噪声！
                    while fixIndex < nextIndex and (sectionArray[fixIndex] - sectionMean) >= yDiffThr:
                        fixedSectionArray[fixIndex] = modelY(zArray[fixIndex])

                        # 修改点云值
                        points[sectionList[fixIndex], 1] = fixedSectionArray[fixIndex]

                        fixIndex += 1
                elif abs(gradientArray[nowIndex]) > 2 * gradientThr:
                    # 若该点只是个跳变的噪声点，且梯度超过2倍的限值，则抹平该点就行
                    if nowIndex == len(sectionList) - 1:
                        fixedSectionArray[nowIndex] = sectionArray[nowIndex - 1]
                    else:
                        fixedSectionArray[nowIndex] = (sectionArray[nowIndex + 1] + sectionArray[nowIndex - 1]) / 2

                    # 修改点云值
                    points[sectionList[nowIndex], 1] = fixedSectionArray[nowIndex]

                    # 如果下一个相邻点就是突变点，则跳过
                    if gradientIndex < len(gradientIndexList) - 1 and gradientIndexList[gradientIndex + 1] == nowIndex + 1:
                        gradientIndex += 1

                gradientIndex += 1
                continue

            # 点的某一侧是正常砟石，某一侧不正常，接下来就是判断哪一侧不正常
            leftDiff = abs(sectionArray[nowIndex - 1] - modelY(zArray[nowIndex - 1]))
            rightDiff = abs(sectionArray[nowIndex] - modelY(zArray[nowIndex]))
            # 左边不正常
            if leftDiff > rightDiff:
                fixIndex = nowIndex - 1
                # 到上一个点为止全是噪声！
                while fixIndex >= lastIndex and (sectionArray[fixIndex] - sectionMean) >= yDiffThr:
                    fixedSectionArray[fixIndex] = modelY(zArray[fixIndex])

                    # 修改点云值
                    points[sectionList[fixIndex], 1] = fixedSectionArray[fixIndex]

                    fixIndex -= 1
            # 右边不正常
            elif leftDiff < rightDiff:
                fixIndex = nowIndex
                # 到下一个点为止全是噪声！
                while fixIndex < nextIndex and (sectionArray[fixIndex] - sectionMean) >= yDiffThr:
                    fixedSectionArray[fixIndex] = modelY(zArray[fixIndex])

                    # 修改点云值
                    points[sectionList[fixIndex], 1] = fixedSectionArray[fixIndex]

                    fixIndex += 1

            gradientIndex += 1

        if show:
            plt.subplot(3, 1, 3)
            plt.plot(zArray, fixedSectionArray)
            plt.axis("equal")
            plt.grid('on')

            if showPlt:
                plt.suptitle(f'Section {mileageIndex}')
                plt.show()

    return showCostTime


@costTime
def windowSlide(points, show=False):
    """
    用行数值进行轨枕分割。
    :param points: (np.array) 需要分割的点云
    :param show: (bool) 是否显示结果
    :return: None
    """
    rowDir = os.path.join(outputDir, "rowScanSleeper")
    os.makedirs(rowDir, exist_ok=True)
    showCostTime = 0.0

    sleeperLength = 1.2
    railOutside = 0.95
    railInside = 0.55

    slopThr = 0.5  # 坡度
    yThr = -0.24  # 最低深度阈值
    heightThr0 = 0.015  # 低于这个值的高度差直接算作噪声
    heightThr1 = 0.04  # 高于这个值的高度差直接算作砟石
    lengthIntervalThr = 0.005  # 尖锐噪声造成的小段不连续的最高容忍值
    lengthRatioThr = 0.4  # 轨枕长度在所在区域的最低比例
    sleeperTotalLength = 2 * (sleeperLength - railOutside) + 2 * railInside

    windowSize = heightThr1 / slopThr
    windowStep = lengthIntervalThr * 2

    # 计算里程信息
    assert points.shape[0] > 0
    mileageList = [[]]
    mileageIndex = 0
    for pointIndex in range(points.shape[0]):
        mileageList[mileageIndex].append(pointIndex)
        if pointIndex < points.shape[0] - 1:
            if points[pointIndex + 1, 0] != points[pointIndex, 0]:
                mileageIndex += 1
                mileageList.append([])

    mileageNum = len(mileageList)
    mileageMeanInterval = abs(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (
            mileageNum - 1)
    if mileageNum > 1:
        logger.info(
            f"里程的排列是否均匀："
            f"{(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (points[mileageList[0][0], 0] - points[mileageList[1][0], 0]) == mileageNum - 1}，"
            f"平均断面间隔为：{mileageMeanInterval}m")
    else:
        logger.error("里程数不足！")
        return showCostTime

    pointColors = np.zeros([points.shape[0], 3])
    pointColors[:] = [0, 255, 0]  # 先为所有点附上绿色

    ifTest = False

    # 首先进行初次筛选
    for mileageIndex, sectionList in tqdm(enumerate(mileageList)):
        if ifTest:
            sectionArrayX = []
            sectionArrayY = []
            borderLeftX = []
            borderLeftY = []
            borderRightX = []
            borderRightY = []
            strangeLeftX = []
            strangeLeftY = []
            strangeRightX = []
            strangeRightY = []
            finalLeftX = []
            finalLeftY = []
            finalRightX = []
            finalRightY = []

        areaIndex = 0  # 当前遍历的区域，0为轨枕左侧，1为钢轨左侧，2为左钢轨，3为钢轨中间，4为右钢轨，5为钢轨右侧，6为轨枕右侧
        areaRightBorder = [-sleeperLength, -railOutside, -railInside, railInside, railOutside, sleeperLength]

        # 先进行初次筛查
        sleeperList = [[] for _ in range(6)]
        borderIndex = 0
        leftDescend = 0
        sectionIndex = 0
        while sectionIndex < len(sectionList):
            if ifTest:
                sectionArrayX.append(points[sectionList[sectionIndex], 2])
                sectionArrayY.append(points[sectionList[sectionIndex], 1])

            pZ = points[sectionList[sectionIndex], 2]
            pY = points[sectionList[sectionIndex], 1]
            # 若在轨枕右侧，则跳出循环
            if areaIndex == 6:
                break
            # 跳出右边界点前做最后一次计算
            elif pZ > areaRightBorder[areaIndex]:
                if borderIndex != sectionIndex - 1 and leftDescend != 0 and areaIndex % 2:
                    height = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[borderIndex], 1])
                    # 必要条件
                    if height <= heightThr1:
                        width = points[sectionList[sectionIndex - 1], 2] - points[sectionList[borderIndex], 2]
                        if width == 0:
                            logger.warning(f'发现了宽度差为0的线段在点{borderIndex}和点{sectionIndex - 1}之间')
                        else:
                            slop = height / width
                            # 充分条件
                            if height <= heightThr0 or slop <= slopThr:
                                sleeperList[areaIndex].append(sectionList[borderIndex])
                                sleeperList[areaIndex].append(sectionList[sectionIndex - 1])
                borderIndex = sectionIndex
                leftDescend = 0
                areaIndex += 1
            # 如果点落在轨枕左侧或钢轨屏蔽区域，则跳过当前点
            elif pZ < -sleeperLength or railInside < abs(pZ) < railOutside:
                borderIndex = sectionIndex + 1
            # 如果点的深度过低，则不是轨枕点，应从此处断开
            elif pY < yThr:
                if sectionIndex - 1 != borderIndex and leftDescend != 0:
                    height = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[borderIndex], 1])
                    # 必要条件
                    if height <= heightThr1:
                        width = points[sectionList[sectionIndex - 1], 2] - points[sectionList[borderIndex], 2]
                        if width == 0:
                            logger.warning(f'发现了宽度差为0的线段在点{borderIndex}和点{sectionIndex - 1}之间')
                        else:
                            slop = height / width
                            # 充分条件
                            if height <= heightThr0 or slop <= slopThr:
                                sleeperList[areaIndex].append(sectionList[borderIndex])
                                sleeperList[areaIndex].append(sectionList[sectionIndex - 1])
                borderIndex = sectionIndex + 1
                leftDescend = 0
                sectionIndex += 1
            else:
                # 计算当前差分值
                rightDescend = points[sectionList[sectionIndex], 1] - points[sectionList[sectionIndex - 1], 1]
                # 当前差分与之前差分乘积为负数，即拐点
                if leftDescend * rightDescend < 0:
                    # 计算两个拐点间的高度差
                    height = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[borderIndex], 1])
                    # 高度差必要条件
                    if height <= heightThr1:
                        width = points[sectionList[sectionIndex - 1], 2] - points[sectionList[borderIndex], 2]
                        if width == 0:
                            logger.warning(f'发现了宽度差为0的线段在点{borderIndex}和点{sectionIndex - 1}之间')
                        else:
                            slop = height / width
                            # 小于高度差充分条件或者坡度在阈值内的才计入轨枕
                            if height <= heightThr0 or slop <= slopThr:
                                sleeperList[areaIndex].append(sectionList[borderIndex])
                                sleeperList[areaIndex].append(sectionList[sectionIndex - 1])
                    borderIndex = sectionIndex - 1
                    leftDescend = rightDescend
                elif leftDescend == 0:
                    leftDescend = rightDescend
            sectionIndex += 1

        finalSleeperList = []
        allSum = 0  # 总线段的长度
        orinSum = 0
        # 根据长度进行二次筛查
        for areaIndex in range(6):
            if areaIndex % 2:
                perSum = 0  # 小线段的长度
                leftBorder = -1  # 上次索引的线段的左端点索引
                rightBorder = -1  # 记录上一段的右端点索引
                for sectionIndex in range(0, len(sleeperList[areaIndex]), 2):
                    leftIndex = sleeperList[areaIndex][sectionIndex]
                    rightIndex = sleeperList[areaIndex][sectionIndex + 1]

                    if ifTest:
                        borderLeftX.append(points[leftIndex, 2])
                        borderRightX.append(points[rightIndex, 2])
                        borderLeftY.append(points[leftIndex, 1])
                        borderRightY.append(points[rightIndex, 1])

                    # 先计算当前一小段的长度
                    thisLength = points[rightIndex, 2] - points[leftIndex, 2]
                    orinSum += thisLength

                    # 若这是计算的第一个线段
                    if leftBorder == -1 and rightBorder == -1:
                        perSum += thisLength
                        leftBorder = leftIndex
                    # 若这是计算的第一个线段，或者距离上一个线段不超过lengthIntervalThr
                    elif points[leftIndex, 2] - points[rightBorder, 2] <= lengthIntervalThr:
                        perSum += thisLength
                    else:
                        # 不是的话，就以滑窗的形式来排除尖峰
                        if perSum > lengthIntervalThr:
                            finalIndex = []
                            windowLeft = leftBorder
                            nextWindow = True
                            while nextWindow:
                                nextWindow = False
                                leftZ = points[windowLeft, 2]
                                rightZ = leftZ + windowSize
                                pIndex = windowLeft
                                minY = float('inf')
                                minX = 0
                                minIndex = 0
                                maxY = float('-inf')
                                maxX = 0
                                maxIndex = 0
                                while pIndex <= rightBorder and points[pIndex, 2] <= rightZ:
                                    if points[pIndex, 1] > maxY:
                                        maxY = points[pIndex, 1]
                                        maxX = points[pIndex, 2]
                                        maxIndex = pIndex
                                    if points[pIndex, 1] < minY:
                                        minY = points[pIndex, 1]
                                        minX = points[pIndex, 2]
                                        minIndex = pIndex
                                    if not nextWindow and points[pIndex, 2] > leftZ + windowStep:
                                        nextWindow = True
                                        windowLeft = pIndex
                                    pIndex += 1

                                if maxX > minX:
                                    leftI = minIndex
                                    rightI = maxIndex
                                else:
                                    leftI = maxIndex
                                    rightI = minIndex

                                width = abs(maxX - minX)
                                if width > 0:
                                    height = maxY - minY
                                    slop = height / width
                                    if height <= heightThr0 or (height <= heightThr1 and slop < slopThr):
                                        pass
                                    else:
                                        perSum -= width
                                        finalIndex.extend([leftI, rightI])
                                        if nextWindow and windowLeft < rightI and points[rightBorder, 2] - points[
                                            rightI, 2] > lengthIntervalThr:
                                            windowLeft = rightI

                                        if ifTest:
                                            strangeLeftX.append(points[leftI, 2])
                                            strangeRightX.append(points[rightI, 2])
                                            strangeLeftY.append(points[leftI, 1])
                                            strangeRightY.append(points[rightI, 1])

                            finalIndex.append(rightBorder)
                            finalIndex.insert(0, leftBorder)

                            for k in range(0, len(finalIndex), 2):
                                width = points[finalIndex[k + 1], 2] - points[finalIndex[k], 2]
                                if width > lengthIntervalThr:
                                    finalSleeperList.extend([finalIndex[k], finalIndex[k + 1]])

                                    if ifTest:
                                        finalLeftX.append(points[finalIndex[k], 2])
                                        finalRightX.append(points[finalIndex[k + 1], 2])
                                        finalLeftY.append(points[finalIndex[k], 1])
                                        finalRightY.append(points[finalIndex[k + 1], 1])
                                else:
                                    perSum -= width

                                    if ifTest:
                                        strangeLeftX.append(points[finalIndex[k], 2])
                                        strangeRightX.append(points[finalIndex[k + 1], 2])
                                        strangeLeftY.append(points[finalIndex[k], 1])
                                        strangeRightY.append(points[finalIndex[k + 1], 1])

                            allSum += perSum

                        else:
                            if ifTest:
                                strangeLeftX.append(points[leftBorder, 2])
                                strangeRightX.append(points[rightBorder, 2])
                                strangeLeftY.append(points[leftBorder, 1])
                                strangeRightY.append(points[rightBorder, 1])

                        leftBorder = leftIndex
                        perSum = thisLength

                    rightBorder = rightIndex

                    # 如果是最后一条线段，则也要进行判断了
                    if sectionIndex == len(sleeperList[areaIndex]) - 2:
                        if perSum > lengthIntervalThr:
                            finalIndex = []
                            windowLeft = leftBorder
                            nextWindow = True
                            while nextWindow:
                                nextWindow = False
                                leftZ = points[windowLeft, 2]
                                rightZ = leftZ + windowSize
                                pIndex = windowLeft
                                minY = float('inf')
                                minX = 0
                                minIndex = 0
                                maxY = float('-inf')
                                maxX = 0
                                maxIndex = 0
                                while pIndex <= rightBorder and points[pIndex, 2] <= rightZ:
                                    if points[pIndex, 1] > maxY:
                                        maxY = points[pIndex, 1]
                                        maxX = points[pIndex, 2]
                                        maxIndex = pIndex
                                    if points[pIndex, 1] < minY:
                                        minY = points[pIndex, 1]
                                        minX = points[pIndex, 2]
                                        minIndex = pIndex
                                    if not nextWindow and points[pIndex, 2] > leftZ + windowStep:
                                        nextWindow = True
                                        windowLeft = pIndex
                                    pIndex += 1

                                if maxX > minX:
                                    leftI = minIndex
                                    rightI = maxIndex
                                elif maxX < minX:
                                    leftI = maxIndex
                                    rightI = minIndex
                                else:
                                    continue

                                width = abs(maxX - minX)
                                if width > 0:
                                    height = maxY - minY
                                    slop = height / width
                                    if height <= heightThr0 or (height <= heightThr1 and slop < slopThr):
                                        pass
                                    else:
                                        perSum -= width
                                        finalIndex.extend([leftIndex, rightI])
                                        if nextWindow and windowLeft < rightI and points[rightBorder, 2] - points[
                                            rightI, 2] > lengthIntervalThr:
                                            windowLeft = rightI

                                        if ifTest:
                                            strangeLeftX.append(points[leftI, 2])
                                            strangeRightX.append(points[rightI, 2])
                                            strangeLeftY.append(points[leftI, 1])
                                            strangeRightY.append(points[rightI, 1])

                            finalIndex.append(rightBorder)
                            finalIndex.insert(0, leftBorder)

                            for k in range(0, len(finalIndex), 2):
                                width = points[finalIndex[k + 1], 2] - points[finalIndex[k], 2]
                                if width > lengthIntervalThr:
                                    finalSleeperList.extend([finalIndex[k], finalIndex[k + 1]])

                                    if ifTest:
                                        finalLeftX.append(points[finalIndex[k], 2])
                                        finalRightX.append(points[finalIndex[k + 1], 2])
                                        finalLeftY.append(points[finalIndex[k], 1])
                                        finalRightY.append(points[finalIndex[k + 1], 1])
                                else:
                                    perSum -= width

                                    if ifTest:
                                        strangeLeftX.append(points[finalIndex[k], 2])
                                        strangeRightX.append(points[finalIndex[k + 1], 2])
                                        strangeLeftY.append(points[finalIndex[k], 1])
                                        strangeRightY.append(points[finalIndex[k + 1], 1])

                            allSum += perSum
                        else:
                            if ifTest:
                                strangeLeftX.append(points[leftBorder, 2])
                                strangeRightX.append(points[rightBorder, 2])
                                strangeLeftY.append(points[leftBorder, 1])
                                strangeRightY.append(points[rightBorder, 1])

        startTime = time.time()
        if ifTest:
            plt.figure(1)
            plt.clf()
            plt.subplot(2, 1, 1)
            plt.plot(sectionArrayX, sectionArrayY, 'g')
            plt.plot(borderLeftX, borderLeftY, 'ro')
            plt.plot(borderRightX, borderRightY, 'b.')
            plt.axis("equal")
            plt.grid('on')
            plt.subplot(2, 1, 2)
            plt.plot(sectionArrayX, sectionArrayY, 'g')
            plt.plot(finalLeftX, finalLeftY, 'ro')
            plt.plot(finalRightX, finalRightY, 'b.')
            plt.plot(strangeLeftX, strangeLeftY, 'm>')
            plt.plot(strangeRightX, strangeRightY, 'c<')
            plt.suptitle(f'Section {mileageIndex}, allSum = {allSum}')
            plt.axis("equal")
            plt.grid('on')
            plt.show()
        endTime = time.time()
        showCostTime += endTime - startTime

        # 再根据总长筛选，如果合长度不多于阈值，则跳过当前断面
        if allSum >= sleeperTotalLength * lengthRatioThr:
            # 最后标注点云
            pointIndex = 0
            while pointIndex < len(finalSleeperList):
                leftIndex = finalSleeperList[pointIndex]
                rightIndex = finalSleeperList[pointIndex + 1]
                for pIndex in range(leftIndex, rightIndex + 1):
                    pointColors[pIndex] = [255, 0, 0]  # 轨枕部分用红色标注
                pointIndex += 2

        logger.info(f"断面{mileageIndex}原长{orinSum}，滑窗后长{allSum}，减去了{orinSum - allSum}")

    if show:
        startTime = time.time()
        showPointCloud(points, pointColors)
        endTime = time.time()
        showCostTime += endTime - startTime

    return showCostTime

mileageCount = 0
bridgeEccentricLen = 0
@costTime
def bridgeCal(points, show=False):
    """
    用行数值进行桥梁分割。
    :param points: (np.array) 需要分割的点云
    :param show: (bool) 是否显示结果
    :return: None
    """
    rowDir = os.path.join(outputDir, "rowScanSleeper")
    os.makedirs(rowDir, exist_ok=True)
    showCostTime = 0.0
    global mileageCount
    global bridgeEccentricLen
    sleeperLength = 1.3
    railOutside = 0.95
    railInside = 0.55

    slopThr = 0.5  # 坡度
    yThr = -0.24  # 最低深度阈值
    heightThr0 = 0.015  # 低于这个值的高度差直接算作噪声
    heightThr1 = 0.04  # 高于这个值的高度差直接算作砟石
    lengthIntervalThr = 0.005  # 尖锐噪声造成的小段不连续的最高容忍值
    lengthRatioThr = 0.4  # 轨枕长度在所在区域的最低比例
    sleeperTotalLength = 2 * (sleeperLength - railOutside) + 2 * railInside
    leftBridgeSlopThr = 0.35
    leftBridgeHeightThr = 0.005
    bridgeLengthMinThr = 0.65
    bridgeLengthMaxThr = 1.2
    bridgeTotalLength = 0.8
    windowSize = heightThr1 / slopThr
    windowStep = lengthIntervalThr * 2
    # 计算里程信息
    assert points.shape[0] > 0
    mileageList = []
    mileageIndex = 0
    AllMileageList = [[]]
    leftBridgeMileageList = []
    rightBridgeMileageList = []
    bridgeMileageIndex = -1
    sleeperMileageIndex = -1
    leftBridgeMileageIndex = -1
    rightBridgeMileageIndex = -1
    startSectionIndex = 0
    endSectionIndex = 0
    for pointIndex in range(points.shape[0]):
        AllMileageList[mileageIndex].append(pointIndex)
        if pointIndex < points.shape[0] - 1:
            if points[pointIndex + 1, 0] != points[pointIndex, 0]:
                mileageIndex += 1
                AllMileageList.append([])
    while startSectionIndex < len(AllMileageList):
        for mileageIndex in range(startSectionIndex, len(AllMileageList)):
            if abs(points[AllMileageList[mileageIndex][0], 0] - points[AllMileageList[startSectionIndex][0], 0]) > 1:
                endSectionIndex = mileageIndex
                break
            else:
                mileageList.append([])
                sleeperMileageIndex += 1
                leftBridgeMileageList.append([])
                leftBridgeMileageIndex += 1
                rightBridgeMileageList.append([])
                rightBridgeMileageIndex += 1    
                for pointIndex in AllMileageList[mileageIndex]:
                    if points[pointIndex, 2] >= scanModel[0][0] and points[pointIndex, 2] <= scanModel[5][0]:
                        mileageList[sleeperMileageIndex].append(pointIndex)
                    elif (points[pointIndex, 2] <= scanModel[0][0] and points[pointIndex, 2] >= -5): 
                        #leftBridgeMileageList[leftBridgeMileageIndex].insert(0, pointIndex)
                        leftBridgeMileageList[leftBridgeMileageIndex].append(pointIndex)
                    elif (points[pointIndex, 2] >= scanModel[5][0] and points[pointIndex, 2] <= 5):
                        rightBridgeMileageList[rightBridgeMileageIndex].append(pointIndex)
            if len(mileageList[sleeperMileageIndex]) == 0:
                mileageList.pop()
                sleeperMileageIndex -= 1
            if len(leftBridgeMileageList[leftBridgeMileageIndex]) == 0:
                leftBridgeMileageList.pop()
                leftBridgeMileageIndex -= 1
            if len(rightBridgeMileageList[rightBridgeMileageIndex]) == 0:
                rightBridgeMileageList.pop()
                rightBridgeMileageIndex -= 1
            if mileageIndex == len(AllMileageList) - 1:
                endSectionIndex = len(AllMileageList)
        startSectionIndex = endSectionIndex

        mileageNum = len(mileageList) # 断面数
        mileageMeanInterval = abs(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (
                mileageNum - 1) # 断面间距
        if mileageNum > 1:
            logger.info(
                f"里程的排列是否均匀："
                f"{(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (points[mileageList[0][0], 0] - points[mileageList[1][0], 0]) == mileageNum - 1}，"
                f"平均断面间隔为：{mileageMeanInterval}m")
        else:
            logger.error("里程数不足！")
            return showCostTime

        pointColors = np.zeros([points.shape[0], 3])
        pointColors[:] = [0, 255, 0]  # 先为所有点附上绿色

        ifTest = False
        sleeperMiddleLen = 0
        sleeperCount = 0
        bridgeLeftLen = 0
        bridgeLeftCount = 0
        bridgeRightLen = 0
        bridgeRightCount = 0
        # 首先进行初次筛选
        for mileageIndex, sectionList in tqdm(enumerate(mileageList)):

            if ifTest:
                sectionArrayX = []
                sectionArrayY = []
                borderLeftX = []
                borderLeftY = []
                borderRightX = []
                borderRightY = []
                strangeLeftX = []
                strangeLeftY = []
                strangeRightX = []
                strangeRightY = []
                finalLeftX = []
                finalLeftY = []
                finalRightX = []
                finalRightY = []

            areaIndex = 0  # 当前遍历的区域，0为轨枕左侧，1为钢轨左侧，2为左钢轨，3为钢轨中间，4为右钢轨，5为钢轨右侧，6为轨枕右侧
            areaRightBorder = [-sleeperLength - 0.1, -railOutside, -railInside, railInside, railOutside, sleeperLength + 0.1]

            # 先进行初次筛查
            sleeperList = [[] for _ in range(6)]
            borderIndex = 0
            leftDescend = 0
            sectionIndex = 0
            while sectionIndex < len(sectionList):
                if ifTest:
                    sectionArrayX.append(points[sectionList[sectionIndex], 2])
                    sectionArrayY.append(points[sectionList[sectionIndex], 1])

                pZ = points[sectionList[sectionIndex], 2]
                pY = points[sectionList[sectionIndex], 1]
                # 若在轨枕右侧，则跳出循环
                if areaIndex == 6:
                    break
                # 跳出右边界点前做最后一次计算
                elif pZ > areaRightBorder[areaIndex]:
                    if borderIndex != sectionIndex - 1 and leftDescend != 0 and areaIndex % 2:
                        height = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[borderIndex], 1])
                        # 必要条件
                        if height <= heightThr1:
                            width = points[sectionList[sectionIndex - 1], 2] - points[sectionList[borderIndex], 2]
                            if width == 0:
                                logger.warning(f'发现了宽度差为0的线段在点{borderIndex}和点{sectionIndex - 1}之间')
                            else:
                                slop = height / width
                                # 充分条件
                                if height <= heightThr0 or slop <= slopThr:
                                    sleeperList[areaIndex].append(sectionList[borderIndex])
                                    sleeperList[areaIndex].append(sectionList[sectionIndex - 1])
                    borderIndex = sectionIndex
                    leftDescend = 0
                    areaIndex += 1
                # 如果点落在轨枕左侧或钢轨屏蔽区域，则跳过当前点
                elif pZ < -sleeperLength - 0.1 or railInside < abs(pZ) < railOutside:
                    borderIndex = sectionIndex + 1
                # 如果点的深度过低，则不是轨枕点，应从此处断开
                elif pY < yThr:
                    if sectionIndex - 1 != borderIndex and leftDescend != 0:
                        height = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[borderIndex], 1])
                        # 必要条件
                        if height <= heightThr1:
                            width = points[sectionList[sectionIndex - 1], 2] - points[sectionList[borderIndex], 2]
                            if width == 0:
                                logger.warning(f'发现了宽度差为0的线段在点{borderIndex}和点{sectionIndex - 1}之间')
                            else:
                                slop = height / width
                                # 充分条件
                                if height <= heightThr0 or slop <= slopThr:
                                    sleeperList[areaIndex].append(sectionList[borderIndex])
                                    sleeperList[areaIndex].append(sectionList[sectionIndex - 1])
                    borderIndex = sectionIndex + 1
                    leftDescend = 0
                    sectionIndex += 1
                else:
                    # 计算当前差分值
                    rightDescend = points[sectionList[sectionIndex], 1] - points[sectionList[sectionIndex - 1], 1]
                    # 当前差分与之前差分乘积为负数，即拐点
                    if leftDescend * rightDescend < 0:
                        # 计算两个拐点间的高度差
                        height = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[borderIndex], 1])
                        # 高度差必要条件
                        if height <= heightThr1:
                            width = points[sectionList[sectionIndex - 1], 2] - points[sectionList[borderIndex], 2]
                            if width == 0:
                                logger.warning(f'发现了宽度差为0的线段在点{borderIndex}和点{sectionIndex - 1}之间')
                            else:
                                slop = height / width
                                # 小于高度差充分条件或者坡度在阈值内的才计入轨枕
                                if height <= heightThr0 or slop <= slopThr:
                                    sleeperList[areaIndex].append(sectionList[borderIndex])
                                    sleeperList[areaIndex].append(sectionList[sectionIndex - 1])
                        borderIndex = sectionIndex - 1
                        leftDescend = rightDescend
                    elif leftDescend == 0:
                        leftDescend = rightDescend
                sectionIndex += 1

            finalSleeperList = []
            allSum = 0  # 总线段的长度
            orinSum = 0
            # 根据长度进行二次筛查
            for areaIndex in range(6):
                if areaIndex % 2:
                    perSum = 0  # 小线段的长度
                    leftBorder = -1  # 上次索引的线段的左端点索引
                    rightBorder = -1  # 记录上一段的右端点索引
                    for sectionIndex in range(0, len(sleeperList[areaIndex]), 2):
                        leftIndex = sleeperList[areaIndex][sectionIndex]
                        rightIndex = sleeperList[areaIndex][sectionIndex + 1]

                        if ifTest:
                            borderLeftX.append(points[leftIndex, 2])
                            borderRightX.append(points[rightIndex, 2])
                            borderLeftY.append(points[leftIndex, 1])
                            borderRightY.append(points[rightIndex, 1])

                        # 先计算当前一小段的长度
                        thisLength = points[rightIndex, 2] - points[leftIndex, 2]
                        orinSum += thisLength

                        # 若这是计算的第一个线段
                        if leftBorder == -1 and rightBorder == -1:
                            perSum += thisLength
                            leftBorder = leftIndex
                        # 若这是计算的第一个线段，或者距离上一个线段不超过lengthIntervalThr
                        elif points[leftIndex, 2] - points[rightBorder, 2] <= lengthIntervalThr:
                            perSum += thisLength
                        else:
                            # 不是的话，就以滑窗的形式来排除尖峰
                            if perSum > lengthIntervalThr:
                                finalIndex = []
                                windowLeft = leftBorder
                                nextWindow = True
                                while nextWindow:
                                    nextWindow = False
                                    leftZ = points[windowLeft, 2]
                                    rightZ = leftZ + windowSize
                                    pIndex = windowLeft
                                    minY = float('inf')
                                    minX = 0
                                    minIndex = 0
                                    maxY = float('-inf')
                                    maxX = 0
                                    maxIndex = 0
                                    while pIndex <= rightBorder and points[pIndex, 2] <= rightZ:
                                        if points[pIndex, 1] > maxY:
                                            maxY = points[pIndex, 1]
                                            maxX = points[pIndex, 2]
                                            maxIndex = pIndex
                                        if points[pIndex, 1] < minY:
                                            minY = points[pIndex, 1]
                                            minX = points[pIndex, 2]
                                            minIndex = pIndex
                                        if not nextWindow and points[pIndex, 2] > leftZ + windowStep:
                                            nextWindow = True
                                            windowLeft = pIndex
                                        pIndex += 1

                                    if maxX > minX:
                                        leftI = minIndex
                                        rightI = maxIndex
                                    else:
                                        leftI = maxIndex
                                        rightI = minIndex

                                    width = abs(maxX - minX)
                                    if width > 0:
                                        height = maxY - minY
                                        slop = height / width
                                        if height <= heightThr0 or (height <= heightThr1 and slop < slopThr):
                                            pass
                                        else:
                                            perSum -= width
                                            finalIndex.extend([leftI, rightI])
                                            if nextWindow and windowLeft < rightI and points[rightBorder, 2] - points[
                                                rightI, 2] > lengthIntervalThr:
                                                windowLeft = rightI

                                            if ifTest:
                                                strangeLeftX.append(points[leftI, 2])
                                                strangeRightX.append(points[rightI, 2])
                                                strangeLeftY.append(points[leftI, 1])
                                                strangeRightY.append(points[rightI, 1])

                                finalIndex.append(rightBorder)
                                finalIndex.insert(0, leftBorder)

                                for k in range(0, len(finalIndex), 2):
                                    width = points[finalIndex[k + 1], 2] - points[finalIndex[k], 2]
                                    if width > lengthIntervalThr:
                                        finalSleeperList.extend([finalIndex[k], finalIndex[k + 1]])

                                        if ifTest:
                                            finalLeftX.append(points[finalIndex[k], 2])
                                            finalRightX.append(points[finalIndex[k + 1], 2])
                                            finalLeftY.append(points[finalIndex[k], 1])
                                            finalRightY.append(points[finalIndex[k + 1], 1])
                                    else:
                                        perSum -= width

                                        if ifTest:
                                            strangeLeftX.append(points[finalIndex[k], 2])
                                            strangeRightX.append(points[finalIndex[k + 1], 2])
                                            strangeLeftY.append(points[finalIndex[k], 1])
                                            strangeRightY.append(points[finalIndex[k + 1], 1])

                                allSum += perSum

                            else:
                                if ifTest:
                                    strangeLeftX.append(points[leftBorder, 2])
                                    strangeRightX.append(points[rightBorder, 2])
                                    strangeLeftY.append(points[leftBorder, 1])
                                    strangeRightY.append(points[rightBorder, 1])

                            leftBorder = leftIndex
                            perSum = thisLength

                        rightBorder = rightIndex

                        # 如果是最后一条线段，则也要进行判断了
                        if sectionIndex == len(sleeperList[areaIndex]) - 2:
                            if perSum > lengthIntervalThr:
                                finalIndex = []
                                windowLeft = leftBorder
                                nextWindow = True
                                while nextWindow:
                                    nextWindow = False
                                    leftZ = points[windowLeft, 2]
                                    rightZ = leftZ + windowSize
                                    pIndex = windowLeft
                                    minY = float('inf')
                                    minX = 0
                                    minIndex = 0
                                    maxY = float('-inf')
                                    maxX = 0
                                    maxIndex = 0
                                    while pIndex <= rightBorder and points[pIndex, 2] <= rightZ:
                                        if points[pIndex, 1] > maxY:
                                            maxY = points[pIndex, 1]
                                            maxX = points[pIndex, 2]
                                            maxIndex = pIndex
                                        if points[pIndex, 1] < minY:
                                            minY = points[pIndex, 1]
                                            minX = points[pIndex, 2]
                                            minIndex = pIndex
                                        if not nextWindow and points[pIndex, 2] > leftZ + windowStep:
                                            nextWindow = True
                                            windowLeft = pIndex
                                        pIndex += 1

                                    if maxX > minX:
                                        leftI = minIndex
                                        rightI = maxIndex
                                    elif maxX < minX:
                                        leftI = maxIndex
                                        rightI = minIndex
                                    else:
                                        continue

                                    width = abs(maxX - minX)
                                    if width > 0:
                                        height = maxY - minY
                                        slop = height / width
                                        if height <= heightThr0 or (height <= heightThr1 and slop < slopThr):
                                            pass
                                        else:
                                            perSum -= width
                                            finalIndex.extend([leftIndex, rightI])
                                            if nextWindow and windowLeft < rightI and points[rightBorder, 2] - points[
                                                rightI, 2] > lengthIntervalThr:
                                                windowLeft = rightI

                                            if ifTest:
                                                strangeLeftX.append(points[leftI, 2])
                                                strangeRightX.append(points[rightI, 2])
                                                strangeLeftY.append(points[leftI, 1])
                                                strangeRightY.append(points[rightI, 1])

                                finalIndex.append(rightBorder)
                                finalIndex.insert(0, leftBorder)

                                for k in range(0, len(finalIndex), 2):
                                    width = points[finalIndex[k + 1], 2] - points[finalIndex[k], 2]
                                    if width > lengthIntervalThr:
                                        finalSleeperList.extend([finalIndex[k], finalIndex[k + 1]])

                                        if ifTest:
                                            finalLeftX.append(points[finalIndex[k], 2])
                                            finalRightX.append(points[finalIndex[k + 1], 2])
                                            finalLeftY.append(points[finalIndex[k], 1])
                                            finalRightY.append(points[finalIndex[k + 1], 1])
                                    else:
                                        perSum -= width

                                        if ifTest:
                                            strangeLeftX.append(points[finalIndex[k], 2])
                                            strangeRightX.append(points[finalIndex[k + 1], 2])
                                            strangeLeftY.append(points[finalIndex[k], 1])
                                            strangeRightY.append(points[finalIndex[k + 1], 1])

                                allSum += perSum
                            else:
                                if ifTest:
                                    strangeLeftX.append(points[leftBorder, 2])
                                    strangeRightX.append(points[rightBorder, 2])
                                    strangeLeftY.append(points[leftBorder, 1])
                                    strangeRightY.append(points[rightBorder, 1])

            startTime = time.time()
            if ifTest:
                plt.figure(1)
                plt.clf()
                plt.subplot(2, 1, 1)
                plt.plot(sectionArrayX, sectionArrayY, 'g')
                plt.plot(borderLeftX, borderLeftY, 'ro')
                plt.plot(borderRightX, borderRightY, 'b.')
                plt.axis("equal")
                plt.grid('on')
                plt.subplot(2, 1, 2)
                plt.plot(sectionArrayX, sectionArrayY, 'g')
                plt.plot(finalLeftX, finalLeftY, 'ro')
                plt.plot(finalRightX, finalRightY, 'b.')
                plt.plot(strangeLeftX, strangeLeftY, 'm>')
                plt.plot(strangeRightX, strangeRightY, 'c<')
                plt.suptitle(f'Section {mileageIndex}, allSum = {allSum}')
                plt.axis("equal")
                plt.grid('on')
                plt.show()
            endTime = time.time()
            showCostTime += endTime - startTime
            
            # 再根据总长筛选，如果合长度不多于阈值，则跳过当前断面
            if allSum >= sleeperTotalLength * lengthRatioThr:
                # 最后标注点云
                pointIndex = 0
                while pointIndex < len(finalSleeperList):
                    leftIndex = finalSleeperList[pointIndex]
                    rightIndex = finalSleeperList[pointIndex + 1]
                    for pIndex in range(leftIndex, rightIndex + 1):
                        pointColors[pIndex] = [255, 0, 0]  # 轨枕部分用红色标注
                    pointIndex += 2
                sleeperMiddleIndex = (finalSleeperList[0] + finalSleeperList[-1]) // 2
                pointColors[sleeperMiddleIndex] = [0, 0, 255]  # 轨枕中间用蓝色标注
                sleeperCount += 1
                sleeperMiddleLen += points[sleeperMiddleIndex, 2]

            logger.info(f"断面{mileageIndex}原长{orinSum}，滑窗后长{allSum}，减去了{orinSum - allSum}")
        
        for bridgeMileageIndex, sectionList in tqdm(enumerate(leftBridgeMileageList)):
            if ifTest:
                sectionArrayX = []
                sectionArrayY = []
                borderLeftX = []
                borderLeftY = []
                borderRightX = []
                borderRightY = []
                strangeLeftX = []
                strangeLeftY = []
                strangeRightX = []
                strangeRightY = []
                finalLeftX = []
                finalLeftY = []
                finalRightX = []
                finalRightY = []

            # 先进行初次筛查
            sleeperList = []
            borderIndex = 0
            leftDescend = 0
            sectionIndex = 1
            
            while sectionIndex < len(sectionList):
                if ifTest:
                    sectionArrayX.append(points[sectionList[sectionIndex], 2])
                    sectionArrayY.append(points[sectionList[sectionIndex], 1])

                pZ = points[sectionList[sectionIndex], 2]
                pY = points[sectionList[sectionIndex], 1]
                # 跳出右边界点前做最后一次计算
                if sectionIndex == len(sectionList) - 1:
                    if borderIndex != sectionIndex - 1:
                        #height1 = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[0], 1])
                        #if height1 < 0.2:
                        height = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[borderIndex], 1])
                        width = points[sectionList[sectionIndex - 1], 2] - points[sectionList[borderIndex], 2]
                        if width == 0:
                            logger.warning(f'发现了宽度差为0的线段在点{borderIndex}和点{sectionIndex - 1}之间')
                        else:
                            slop = height / width
                            # 充分条件
                            if height <= leftBridgeHeightThr and slop <= leftBridgeSlopThr:
                                sleeperList.append(sectionList[borderIndex])
                                sleeperList.append(sectionList[sectionIndex - 1])
                        #height1 = abs(points[sectionList[sectionIndex], 1] - points[sectionList[0], 1])
                        #if height1 < 0.2:
                        height = abs(points[sectionList[sectionIndex], 1] - points[sectionList[sectionIndex - 1], 1])
                        width = points[sectionList[sectionIndex], 2] - points[sectionList[sectionIndex - 1], 2]
                        if width == 0:
                            logger.warning(f'发现了宽度差为0的线段在点{borderIndex}和点{sectionIndex - 1}之间')
                        else:
                            slop = height / width
                            # 充分条件
                            if height <= leftBridgeHeightThr and slop <= leftBridgeSlopThr:
                                sleeperList.append(sectionList[sectionIndex - 1])
                                sleeperList.append(sectionList[sectionIndex])
                    borderIndex = sectionIndex
                # 如果点的深度过高，则不是桥梁点，应从此处断开
                elif pY > -0.5:
                    if sectionIndex - 1 != borderIndex:
                        #height1 = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[0], 1])
                        #if height1 < 0.2:
                        height = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[borderIndex], 1])
                        width = points[sectionList[sectionIndex - 1], 2] - points[sectionList[borderIndex], 2]
                        if width == 0:
                            logger.warning(f'发现了宽度差为0的线段在点{borderIndex}和点{sectionIndex - 1}之间')
                        else:
                            slop = height / width
                            # 充分条件
                            if height <= leftBridgeHeightThr and slop <= leftBridgeSlopThr:
                                sleeperList.append(sectionList[borderIndex])
                                sleeperList.append(sectionList[sectionIndex - 1])
                    borderIndex = sectionIndex + 1
                    sectionIndex += 1
                else:
                    #height1 = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[0], 1])
                    #if height1 < 0.2:
                    height = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[borderIndex], 1])
                    width = points[sectionList[sectionIndex - 1], 2] - points[sectionList[borderIndex], 2]
                    if width == 0:
                        logger.warning(f'发现了宽度差为0的线段在点{borderIndex}和点{sectionIndex - 1}之间')
                    else:
                        slop = height / width
                        # 充分条件
                        if height <= leftBridgeHeightThr and slop <= leftBridgeSlopThr:
                            sleeperList.append(sectionList[borderIndex])
                            sleeperList.append(sectionList[sectionIndex - 1])
                    borderIndex = sectionIndex - 1
                sectionIndex += 1
            
            allSum = 0  # 总线段的长度
            orinSum = 0
            # 根据长度进行二次筛查
            perSum = 0  # 小线段的长度
            leftBorder = -1  # 上次索引的线段的左端点索引
            rightBorder = -1  # 记录上一段的右端点索引
            finalIndex = []
            for sectionIndex in range(0, len(sleeperList), 2):
            #for sectionIndex in range(len(sleeperList) - 1, -1, -2):
                leftIndex = sleeperList[sectionIndex]
                rightIndex = sleeperList[sectionIndex + 1]

                if ifTest:
                    borderLeftX.append(points[leftIndex, 2])
                    borderRightX.append(points[rightIndex, 2])
                    borderLeftY.append(points[leftIndex, 1])
                    borderRightY.append(points[rightIndex, 1])

                # 先计算当前一小段的长度
                thisLength = points[rightIndex, 2] - points[leftIndex, 2]
                orinSum += thisLength

                # 若这是计算的第一个线段
                if leftBorder == -1 and rightBorder == -1:
                    perSum += thisLength
                    #height1 = (points[rightIndex, 1] + points[leftIndex, 1]) / 2
                    leftBorder = leftIndex
                # 若这是计算的第一个线段，或者距离上一个线段不超过lengthIntervalThr
                elif points[leftIndex, 2] - points[rightBorder, 2] <= lengthIntervalThr:
                    #height = (points[rightIndex, 1] + points[leftIndex, 1]) / 2
                    #if abs(height - height1) < 0.1:
                    perSum += thisLength
                    '''
                    else:
                        if perSum > lengthIntervalThr:
                            finalIndex.append(leftBorder)
                            finalIndex.append(rightBorder)
                            allSum += perSum
                        else:
                            if ifTest:
                                strangeLeftX.append(points[leftBorder, 2])
                                strangeRightX.append(points[rightBorder, 2])
                                strangeLeftY.append(points[leftBorder, 1])
                                strangeRightY.append(points[rightBorder, 1])
                        leftBorder = leftIndex
                        perSum = thisLength'''
                else:
                    # 不是的话，就以滑窗的形式来排除尖峰
                    if perSum > lengthIntervalThr:
                        finalIndex.append(leftBorder)
                        finalIndex.append(rightBorder)
                        allSum += perSum
                    else:
                        if ifTest:
                            strangeLeftX.append(points[leftBorder, 2])
                            strangeRightX.append(points[rightBorder, 2])
                            strangeLeftY.append(points[leftBorder, 1])
                            strangeRightY.append(points[rightBorder, 1])

                    leftBorder = leftIndex
                    perSum = thisLength

                rightBorder = rightIndex

                # 如果是最后一条线段，则也要进行判断了
                if sectionIndex == len(sleeperList) - 2:
                #if sectionIndex == 1:
                    if perSum > lengthIntervalThr:
                        finalIndex.append(leftBorder)
                        finalIndex.append(rightBorder)
                        allSum += perSum   
                    else:
                        if ifTest:
                            strangeLeftX.append(points[leftBorder, 2])
                            strangeRightX.append(points[rightBorder, 2])
                            strangeLeftY.append(points[leftBorder, 1])
                            strangeRightY.append(points[rightBorder, 1])

            startTime = time.time()
            if ifTest:
                plt.figure(1)
                plt.clf()
                plt.subplot(2, 1, 1)
                plt.plot(sectionArrayX, sectionArrayY, 'g')
                plt.plot(borderLeftX, borderLeftY, 'ro')
                plt.plot(borderRightX, borderRightY, 'b.')
                plt.axis("equal")
                plt.grid('on')
                plt.subplot(2, 1, 2)
                plt.plot(sectionArrayX, sectionArrayY, 'g')
                plt.plot(finalLeftX, finalLeftY, 'ro')
                plt.plot(finalRightX, finalRightY, 'b.')
                plt.plot(strangeLeftX, strangeLeftY, 'm>')
                plt.plot(strangeRightX, strangeRightY, 'c<')
                plt.suptitle(f'Section {bridgeMileageIndex}, allSum = {allSum}')
                plt.axis("equal")
                plt.grid('on')
                plt.show()
            endTime = time.time()
            showCostTime += endTime - startTime
            
            # 再根据总长筛选，如果合长度不多于阈值，则跳过当前断面
            if allSum >= bridgeTotalLength * bridgeLengthMinThr and allSum <= bridgeTotalLength * bridgeLengthMaxThr:
                # 最后标注点云
                pointIndex = 0
                while pointIndex < len(finalIndex):
                    leftIndex = finalIndex[pointIndex]
                    rightIndex = finalIndex[pointIndex + 1]
                    for pIndex in range(leftIndex, rightIndex + 1):
                        pointColors[pIndex] = [0, 0, 255]  # 桥梁部分用蓝色标注
                    pointIndex += 2
                bridgeLeftCount += 1
                bridgeLeftLen += points[finalIndex[-1], 2]

            logger.info(f"桥梁左边，断面{bridgeMileageIndex}")

        for bridgeMileageIndex, sectionList in tqdm(enumerate(rightBridgeMileageList)):
            if ifTest:
                sectionArrayX = []
                sectionArrayY = []
                borderLeftX = []
                borderLeftY = []
                borderRightX = []
                borderRightY = []
                strangeLeftX = []
                strangeLeftY = []
                strangeRightX = []
                strangeRightY = []
                finalLeftX = []
                finalLeftY = []
                finalRightX = []
                finalRightY = []

            # 先进行初次筛查
            sleeperList = []
            borderIndex = 0
            leftDescend = 0
            sectionIndex = 1
            while sectionIndex < len(sectionList):
                if ifTest:
                    sectionArrayX.append(points[sectionList[sectionIndex], 2])
                    sectionArrayY.append(points[sectionList[sectionIndex], 1])

                pZ = points[sectionList[sectionIndex], 2]
                pY = points[sectionList[sectionIndex], 1]
                # 跳出右边界点前做最后一次计算
                if sectionIndex == len(sectionList) - 1:
                    if borderIndex != sectionIndex - 1:
                        #height1 = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[0], 1])
                        #if height1 < 0.2:
                        height = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[borderIndex], 1])
                        width = points[sectionList[sectionIndex - 1], 2] - points[sectionList[borderIndex], 2]
                        if width == 0:
                            logger.warning(f'发现了宽度差为0的线段在点{borderIndex}和点{sectionIndex - 1}之间')
                        else:
                            slop = height / width
                            # 充分条件
                            if height <= leftBridgeHeightThr and slop <= leftBridgeSlopThr:
                                sleeperList.append(sectionList[borderIndex])
                                sleeperList.append(sectionList[sectionIndex - 1])
                        #height1 = abs(points[sectionList[sectionIndex], 1] - points[sectionList[0], 1])
                        #if height1 < 0.2:
                        height = abs(points[sectionList[sectionIndex], 1] - points[sectionList[sectionIndex - 1], 1])
                        width = points[sectionList[sectionIndex], 2] - points[sectionList[sectionIndex - 1], 2]
                        if width == 0:
                            logger.warning(f'发现了宽度差为0的线段在点{borderIndex}和点{sectionIndex - 1}之间')
                        else:
                            slop = height / width
                            # 充分条件
                            if height <= leftBridgeHeightThr and slop <= leftBridgeSlopThr:
                                sleeperList.append(sectionList[sectionIndex - 1])
                                sleeperList.append(sectionList[sectionIndex])
                    borderIndex = sectionIndex
                # 如果点的深度过低，则不是轨枕点，应从此处断开
                elif pY > -0.2:
                    if sectionIndex - 1 != borderIndex:
                        #height1 = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[0], 1])
                        #if height1 < 0.2:
                        height = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[borderIndex], 1])
                        width = points[sectionList[sectionIndex - 1], 2] - points[sectionList[borderIndex], 2]
                        if width == 0:
                            logger.warning(f'发现了宽度差为0的线段在点{borderIndex}和点{sectionIndex - 1}之间')
                        else:
                            slop = height / width
                            # 充分条件
                            if height <= leftBridgeHeightThr and slop <= leftBridgeSlopThr:
                                sleeperList.append(sectionList[borderIndex])
                                sleeperList.append(sectionList[sectionIndex - 1])
                    borderIndex = sectionIndex + 1
                    sectionIndex += 1
                else:
                    #height1 = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[0], 1])
                    #if height1 < 0.2:
                    height = abs(points[sectionList[sectionIndex - 1], 1] - points[sectionList[borderIndex], 1])
                    width = points[sectionList[sectionIndex - 1], 2] - points[sectionList[borderIndex], 2]
                    if width == 0:
                        logger.warning(f'发现了宽度差为0的线段在点{borderIndex}和点{sectionIndex - 1}之间')
                    else:
                        slop = height / width
                        # 充分条件
                        if height <= leftBridgeHeightThr and slop <= leftBridgeSlopThr:
                            sleeperList.append(sectionList[borderIndex])
                            sleeperList.append(sectionList[sectionIndex - 1])
                    borderIndex = sectionIndex - 1
                sectionIndex += 1
            
            allSum = 0  # 总线段的长度
            orinSum = 0
            # 根据长度进行二次筛查
            perSum = 0  # 小线段的长度
            leftBorder = -1  # 上次索引的线段的左端点索引
            rightBorder = -1  # 记录上一段的右端点索引
            finalIndex = []
            for sectionIndex in range(0, len(sleeperList), 2):
                leftIndex = sleeperList[sectionIndex]
                rightIndex = sleeperList[sectionIndex + 1]

                if ifTest:
                    borderLeftX.append(points[leftIndex, 2])
                    borderRightX.append(points[rightIndex, 2])
                    borderLeftY.append(points[leftIndex, 1])
                    borderRightY.append(points[rightIndex, 1])

                # 先计算当前一小段的长度
                thisLength = points[rightIndex, 2] - points[leftIndex, 2]
                orinSum += thisLength

                # 若这是计算的第一个线段
                if leftBorder == -1 and rightBorder == -1:
                    perSum += thisLength
                    #height1 = (points[rightIndex, 1] + points[leftIndex, 1]) / 2
                    leftBorder = leftIndex
                # 若这是计算的第一个线段，或者距离上一个线段不超过lengthIntervalThr
                elif points[leftIndex, 2] - points[rightBorder, 2] <= lengthIntervalThr:
                    #height = height1 = (points[rightIndex, 1] + points[leftIndex, 1]) / 2
                    #if abs(height - height1) < 0.1:
                    perSum += thisLength
                    '''
                    else:
                        if perSum > lengthIntervalThr:
                            finalIndex.append(leftBorder)
                            finalIndex.append(rightBorder)
                            allSum += perSum
                        else:
                            if ifTest:
                                strangeLeftX.append(points[leftBorder, 2])
                                strangeRightX.append(points[rightBorder, 2])
                                strangeLeftY.append(points[leftBorder, 1])
                                strangeRightY.append(points[rightBorder, 1])
                        leftBorder = leftIndex
                        perSum = thisLength'''
                else:
                    # 不是的话，就以滑窗的形式来排除尖峰
                    if perSum > lengthIntervalThr:
                        finalIndex.append(leftBorder)
                        finalIndex.append(rightBorder)
                        allSum += perSum
                    else:
                        if ifTest:
                            strangeLeftX.append(points[leftBorder, 2])
                            strangeRightX.append(points[rightBorder, 2])
                            strangeLeftY.append(points[leftBorder, 1])
                            strangeRightY.append(points[rightBorder, 1])

                    leftBorder = leftIndex
                    perSum = thisLength
                rightBorder = rightIndex

                # 如果是最后一条线段，则也要进行判断了
                if sectionIndex == len(sleeperList) - 2:
                    if perSum > lengthIntervalThr:
                        finalIndex.append(leftBorder)
                        finalIndex.append(rightBorder)
                        allSum += perSum
                        
                    else:
                        if ifTest:
                            strangeLeftX.append(points[leftBorder, 2])
                            strangeRightX.append(points[rightBorder, 2])
                            strangeLeftY.append(points[leftBorder, 1])
                            strangeRightY.append(points[rightBorder, 1])

            startTime = time.time()
            if ifTest:
                plt.figure(1)
                plt.clf()
                plt.subplot(2, 1, 1)
                plt.plot(sectionArrayX, sectionArrayY, 'g')
                plt.plot(borderLeftX, borderLeftY, 'ro')
                plt.plot(borderRightX, borderRightY, 'b.')
                plt.axis("equal")
                plt.grid('on')
                plt.subplot(2, 1, 2)
                plt.plot(sectionArrayX, sectionArrayY, 'g')
                plt.plot(finalLeftX, finalLeftY, 'ro')
                plt.plot(finalRightX, finalRightY, 'b.')
                plt.plot(strangeLeftX, strangeLeftY, 'm>')
                plt.plot(strangeRightX, strangeRightY, 'c<')
                plt.suptitle(f'Section {bridgeMileageIndex}, allSum = {allSum}')
                plt.axis("equal")
                plt.grid('on')
                plt.show()
            endTime = time.time()
            showCostTime += endTime - startTime

            # 再根据总长筛选，如果合长度不多于阈值，则跳过当前断面
            if allSum >= bridgeTotalLength * bridgeLengthMinThr and allSum <= bridgeTotalLength * bridgeLengthMaxThr:
                # 最后标注点云
                pointIndex = 0
                while pointIndex < len(finalIndex):
                    leftIndex = finalIndex[pointIndex]
                    rightIndex = finalIndex[pointIndex + 1]
                    for pIndex in range(leftIndex, rightIndex + 1):
                        pointColors[pIndex] = [0, 0, 255]  # 桥梁部分用蓝色标注
                    pointIndex += 2
                bridgeRightCount += 1
                bridgeRightLen += points[finalIndex[-1], 2]
            logger.info(f"桥梁右边，断面{bridgeMileageIndex}")
        if bridgeLeftCount != 0 and bridgeRightCount != 0:
            mileageCount += 1
            sleeperMiddleLen = sleeperMiddleLen / sleeperCount
            sleeperMiddleZ[SlideCount].append(sleeperMiddleLen)
            bridgeLeftLen = bridgeLeftLen / bridgeLeftCount
            bridgeLeftZ[SlideCount].append(bridgeLeftLen)
            bridgeRightLen = bridgeRightLen / bridgeRightCount
            bridgeRightZ[SlideCount].append(bridgeRightLen)
            bridgeEccentricLen  += (abs(bridgeLeftLen - sleeperMiddleLen) - abs(bridgeRightLen - sleeperMiddleLen)) / 2

    if show:
        startTime = time.time()
        showPointCloud(points, pointColors, save = True)
        endTime = time.time()
        showCostTime += endTime - startTime
    return showCostTime

@costTime
def seg2d(points, sectionInterval=5e-3, show=False):
    """
    用mmsegmentation的神经网络做图像语义分割。
    :param points: (np.array) 需要分割的点云
    :param sectionInterval: (float) 横截断面内点云的采样间隔（单位：m）
    :param show: (bool) 是否显示结果
    :return: None
    """
    seg2dDir = os.path.join(outputDir, "seg2d")
    os.makedirs(seg2dDir, exist_ok=True)

    showCostTime = 0.0

    # 计算里程信息
    assert points.shape[0] > 0
    mileageList = [[]]
    mileageIndex = 0
    for pointIndex in range(points.shape[0]):
        mileageList[mileageIndex].append(pointIndex)
        if pointIndex < points.shape[0] - 1:
            if points[pointIndex + 1, 0] != points[pointIndex, 0]:
                mileageIndex += 1
                mileageList.append([])

    mileageNum = len(mileageList)
    mileageMeanInterval = abs(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (
            mileageNum - 1)
    if mileageNum > 1:
        logger.info(
            f"里程的排列是否均匀："
            f"{(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (points[mileageList[0][0], 0] - points[mileageList[1][0], 0]) == mileageNum - 1}，"
            f"平均断面间隔为：{mileageMeanInterval}m")
    else:
        logger.error("里程数不足！")
        return showCostTime

    # 生成差分图
    _, _, _, normGrayImage, absGrayImage, deepGrayImage, image2points = generateDiffImage(
        points, mileageList, mileageNum, mileageMeanInterval, scanModel[2][0], scanModel[3][0],
        sectionInterval)

    if show:
        startTime = time.time()
        cv2.imshow("normGrayImage", normGrayImage)
        cv2.imshow("absGrayImage", absGrayImage)
        cv2.imshow("deepGrayImage", deepGrayImage)
        cv2.waitKey(0)
        endTime = time.time()
        showCostTime += endTime - startTime
    cv2.imwrite(os.path.join(seg2dDir, "normGrayImage.jpg"), normGrayImage)
    cv2.imwrite(os.path.join(seg2dDir, "absGrayImage.jpg"), absGrayImage)
    cv2.imwrite(os.path.join(seg2dDir, "deepGrayImage.jpg"), deepGrayImage)

    config = "../others/mmsegmentation/configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py"
    checkpoint = ("../others/mmsegmentation/configs/pidnet/"
                  "pidnet-l_2xb6-120k_1024x1024-cityscapes_20230303_114514-0783ca6b.pth")
    device = "cuda:0"
    img = np.dstack((normGrayImage.T, absGrayImage.T, deepGrayImage.T))
    # img = deepGrayImage

    if show:
        startTime = time.time()
        cv2.imshow("img", img)
        cv2.waitKey(0)
        endTime = time.time()
        showCostTime += endTime - startTime
    cv2.imwrite(os.path.join(seg2dDir, "img.jpg"), img)

    cropImg = img[:, :1042, :].copy()

    title = "deepGrayImage"
    opacity = 0.5
    outFile = os.path.join(seg2dDir, 'segImage.jpg')
    # build the model from a config file and a checkpoint file
    model = init_model(config, checkpoint, device=device)

    startTime = time.time()
    # test a single image
    result = inference_model(model, cropImg)
    endTime = time.time()
    logger.info(f"2d图像分割网络推理耗时：{endTime - startTime}s")

    startTime = time.time()
    # show the results
    show_result_pyplot(
        model,
        cropImg,
        result,
        title=title,
        opacity=opacity,
        draw_gt=False,
        show=show,
        out_file=outFile)
    endTime = time.time()
    showCostTime += endTime - startTime

    return showCostTime


@costTime
def seg3d(inputPoints, show=False):
    """
    用mmdetection3d的神经网络做点云分割。目前只支持pointnet++和paconv。
    :param inputPoints: (np.array) 需要分割的点云
    :param show: (bool) 是否显示结果
    :return: None
    """
    seg3dDir = os.path.join(outputDir, "seg3d")

    initArgs = {
        'model': '../others/mmdetection3d/configs/paconv/paconv_ssg_8xb8-cosine-150e_s3dis-seg.py',
        'weights': '../others/mmdetection3d/models/'
                   'paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class_20210729_200615-2147b2d1.pth',
        'device': 'cuda:0'
    }
    callArgs = {
        'inputs': {'points': './demo.bin'},
        'out_dir': seg3dDir,
        'show': show,
        'wait_time': -1,
        'no_save_vis': False,
        'no_save_pred': False,
        'print_result': False
    }

    inputPoints = inputPoints.copy()
    # 将点云的数值变为正数
    inputPoints[:, 0] -= min(np.min(inputPoints[:, 0]), 0)
    inputPoints[:, 1] -= min(np.min(inputPoints[:, 1]), 0)
    inputPoints[:, 2] -= min(np.min(inputPoints[:, 2]), 0)

    # 将点云保存为bin文件（目前mmdetection3d还没直接支持读取np数据，因此得先将其转为bin数据）
    binFilePath = './demo.bin'
    # 为了适应pacconv的数据格式，额外补充3个RGB维度
    binPoints = np.zeros([inputPoints.shape[0], 6], dtype=np.float32)
    binPoints[:, 0] = inputPoints[:, 0].copy()
    binPoints[:, 1] = inputPoints[:, 1].copy()
    binPoints[:, 2] = inputPoints[:, 2].copy()
    # if not os.path.exists(binFilePath):
    # 注意，将二进制数写进bin文件前，一定要先开辟出一个空间存储points的副本，
    # 不然由于之前的sort操作，指针会乱，进而导致bin文件读取的顺序也是乱的！
    with open(binFilePath, 'wb') as f:
        f.write(binPoints.tobytes())

    # 推理
    startTime = time.time()
    seg3dInference = LidarSeg3DInferencer(**initArgs)
    seg3dInference(**callArgs)
    endTime = time.time()
    logger.info(f"3d点云分割神经网络用时{endTime - startTime}s")
    return 0.0


@costTime
def method2d(points, sectionInterval=5e-3, show=False):
    """
    基于二维图像边缘特征的分割方法。
    :param points: (np.array[n, 4]) 输入的点云
    :param sectionInterval: (float) 横截断面内点云的采样间隔（单位：m）
    :param show: (bool) 是否显示结果
    :return: None
    """
    # 先设定计算结果保存目录
    method2dDir = os.path.join(outputDir, "method2d")
    os.makedirs(method2dDir, exist_ok=True)

    showCostTime = 0.0

    # 计算里程信息
    assert points.shape[0] > 0
    mileageList = [[]]
    mileageIndex = 0
    for pointIndex in range(points.shape[0]):
        mileageList[mileageIndex].append(pointIndex)
        if pointIndex < points.shape[0] - 1:
            if points[pointIndex + 1, 0] != points[pointIndex, 0]:
                mileageIndex += 1
                mileageList.append([])

    mileageNum = len(mileageList)
    mileageMeanInterval = abs(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (
            mileageNum - 1)
    if mileageNum > 1:
        logger.info(
            f"里程的排列是否均匀："
            f"{(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (points[mileageList[0][0], 0] - points[mileageList[1][0], 0]) == mileageNum - 1}，"
            f"平均断面间隔为：{mileageMeanInterval}m")
    else:
        logger.error("里程数不足！")
        return showCostTime

    # 生成差分图
    _, _, _, normGrayImage, absGrayImage, deepGrayImage, image2points = generateDiffImage(
        points, mileageList, mileageNum, mileageMeanInterval, scanModel[2][0], scanModel[3][0],
        sectionInterval)

    if show:
        startTime = time.time()
        cv2.imshow(f'normGrayImage', normGrayImage)
        cv2.imshow(f'absGrayImage', absGrayImage)
        cv2.imshow('deepGrayImage', deepGrayImage)
        endTime = time.time()
        showCostTime += endTime - startTime
        cv2.waitKey(0)
    cv2.imwrite(os.path.join(method2dDir, 'normGrayImage.jpg'), normGrayImage)
    cv2.imwrite(os.path.join(method2dDir, 'absGrayImage.jpg'), absGrayImage)
    cv2.imwrite(os.path.join(method2dDir, 'deepGrayImage.jpg'), deepGrayImage)

    # 中值滤波去除噪声
    initialImages = deepGrayImage.copy()
    medianImage = cv2.medianBlur(initialImages, 5)
    if show:
        startTime = time.time()
        cv2.imshow('medianImage', medianImage)
        cv2.waitKey(0)
        endTime = time.time()
        showCostTime += endTime - startTime

    # 应用Canny边缘检测提取边缘，获取中间平滑的枕木
    cannyImage = cv2.Canny(medianImage, 20, 20)
    if show:
        startTime = time.time()
        cv2.imshow('canny', cannyImage)
        cv2.waitKey(0)
        endTime = time.time()
        showCostTime += endTime - startTime
    cv2.imwrite(os.path.join(method2dDir, 'canny.jpg'), cannyImage)

    # 图像取反
    cannyImage = 255 - cannyImage
    if show:
        startTime = time.time()
        cv2.imshow('inverseCanny', cannyImage)
        cv2.waitKey(0)
        endTime = time.time()
        showCostTime += endTime - startTime
    cv2.imwrite(os.path.join(method2dDir, 'inverseCanny.jpg'), cannyImage)
    # 对分割图像进行腐蚀
    kernel = np.ones((5, 5), np.uint8)  # 设置kernel大小
    erosion = cv2.erode(cannyImage, kernel, iterations=1)
    if show:
        startTime = time.time()
        cv2.imshow("erosion", erosion)
        cv2.waitKey(0)
        endTime = time.time()
        showCostTime += endTime - startTime
    kernel2 = np.ones((3, 3), np.uint8)  # 设置kernel大小
    erosion2 = cv2.erode(erosion, kernel2, iterations=1)
    if show:
        startTime = time.time()
        cv2.imshow("erosion2", erosion2)
        cv2.waitKey(0)
        endTime = time.time()
        showCostTime += endTime - startTime
    cv2.imwrite(os.path.join(method2dDir, "erosion2.jpg"), erosion2)

    # 对腐蚀后的图片进行连通域计算，分割出枕木
    filterSegImage, subSegImages = connectedDomainCompute(erosion2, minArea=400, minIoU=0.5)
    if show:
        startTime = time.time()
        cv2.imshow('filterSegImage', filterSegImage)
        cv2.waitKey(0)
        endTime = time.time()
        showCostTime += endTime - startTime
    cv2.imwrite(os.path.join(method2dDir, "filterSegImage.jpg"), filterSegImage)

    # 做腐蚀的反向操作膨胀，以使得粗定位区域扩大
    dilateSegImage = cv2.dilate(filterSegImage, kernel2, iterations=1)
    dilateSegImage = cv2.dilate(dilateSegImage, kernel, iterations=1)
    if show:
        startTime = time.time()
        cv2.imshow('dilateSegImage', dilateSegImage)
        cv2.waitKey(0)
        endTime = time.time()
        showCostTime += endTime - startTime
    cv2.imwrite(os.path.join(method2dDir, "dilateSegImage.jpg"), dilateSegImage)

    # 最后查看分割的效果
    mask = np.where(dilateSegImage > 0, 1, 0).astype(np.uint8)
    finalSegImage = mask * 255
    if show:
        startTime = time.time()
        cv2.imshow('finalSegImage', finalSegImage)
        cv2.waitKey(0)
        endTime = time.time()
        showCostTime += endTime - startTime
    cv2.imwrite(os.path.join(method2dDir, 'finalSegImage.jpg'), finalSegImage)

    # 提取出对应的粗定位点云
    pointColors = np.zeros([points.shape[0], 3])
    pointColors[:] = [0, 255, 0]  # 先为所有点附上绿色
    for colIndex in range(finalSegImage.shape[1]):
        for rowIndex in range(finalSegImage.shape[0]):
            for pIndex in image2points[rowIndex][colIndex]:
                if finalSegImage[rowIndex, colIndex] == 255:
                    pointColors[pIndex] = [255, 0, 0]  # 轨枕点赋值红色
    if show:
        startTime = time.time()
        showPointCloud(points, colors=pointColors)
        endTime = time.time()
        showCostTime += endTime - startTime
    return showCostTime


def statisticalSeg(points, sectionInterval=5e-3, show=False):
    """
    基于一维统计信号的分割方法。
    :param points: (np.array[n, 4]) 输入的点云
    :param sectionInterval: (float) 横截断面内点云的采样间隔（单位：m）
    :param show: (bool) 是否展示结果
    :return: None
    """
    # 先设定计算结果保存目录
    segDir = os.path.join(outputDir, "statisticalSeg")
    os.makedirs(segDir, exist_ok=True)

    sleeperLength = 1.2
    railOutside = 0.95
    railInside = 0.55

    showCostTime = 0.0

    # 计算里程信息
    assert points.shape[0] > 0
    mileageList = [[]]
    mileageIndex = 0
    for pointIndex in range(points.shape[0]):
        mileageList[mileageIndex].append(pointIndex)
        if pointIndex < points.shape[0] - 1:
            if points[pointIndex + 1, 0] != points[pointIndex, 0]:
                mileageIndex += 1
                mileageList.append([])

    mileageNum = len(mileageList)
    mileageMeanInterval = abs(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (
            mileageNum - 1)
    if mileageNum > 1:
        logger.info(
            f"里程的排列是否均匀：{(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (points[mileageList[0][0], 0] - points[mileageList[1][0], 0]) == mileageNum - 1}，"
            f"平均断面间隔为：{mileageMeanInterval}m")
    else:
        logger.error("里程数不足！")
        return showCostTime

    timeStr = datetime.utcnow().strftime('%H%M%S_%f')[:-3]

    # 轨枕区域划分
    sleeperRange = [[-sleeperLength, -railOutside],
                    [-railInside, railInside],
                    [railOutside, sleeperLength]]

    # 然后，逐个对左中右的轨枕区域做均值行投影
    sleeperRowScore = np.zeros((mileageNum,))
    rowMeanInfo = np.ones((mileageNum, 3)) * scanModel[2][1]
    for mileageIndex, sectionList in enumerate(mileageList):
        areaIndex = -1
        areaNum = 0
        for sectionIndex, pointIndex in enumerate(sectionList):
            if sleeperRange[0][0] <= points[pointIndex, 2] <= sleeperRange[0][1]:
                areaIndex = 0
            elif sleeperRange[1][0] <= points[pointIndex, 2] <= sleeperRange[1][1]:
                if areaIndex != 1 and areaNum != 0:
                    rowMeanInfo[mileageIndex, 0] /= areaNum
                    areaNum = 0
                areaIndex = 1
            elif sleeperRange[2][0] <= points[pointIndex, 2] <= sleeperRange[2][1]:
                if areaIndex != 2 and areaNum != 0:
                    rowMeanInfo[mileageIndex, 1] /= areaNum
                    areaNum = 0
                areaIndex = 2
            elif points[pointIndex, 2] > sleeperRange[2][1]:
                if areaNum != 0:
                    rowMeanInfo[mileageIndex, 2] /= areaNum
                break
            else:
                continue

            if points[pointIndex, 1] > scanModel[0][1]:
                rowMeanInfo[mileageIndex, areaIndex] += points[pointIndex, 1]
                areaNum += 1

        if areaNum == 0:
            plt.figure(1)
            plt.clf()
            pointsY = np.zeros(len(sectionList))
            pointsZ = np.zeros(len(sectionList))
            for sectionIndex, pointIndex in enumerate(sectionList):
                pointsY[sectionIndex] = points[pointIndex, 1]
                pointsZ[sectionIndex] = points[pointIndex, 2]
            plt.plot(pointsZ, pointsY)

    plt.figure(2)
    plt.clf()
    for areaIndex in range(3):
        rowMean = rowMeanInfo[:, areaIndex]
        # rowMeanMean = np.mean(rowMean)
        plt.subplot(3, 1, areaIndex + 1)
        plt.plot(rowMean)
        # plt.plot(np.ones_like(rowMean) * rowMeanMean, '--')

        # 提取波峰，峰宽不做约束，但距离两峰之间距离必须大于一个轨枕的断面数
        # peaks, peaksInfo = find_peaks(rowMeanInfo, height=np.mean(rowMeanInfo), width=0,
        #                               distance=sleeperWidth / mileageMeanInterval)
        # AMPD算法
        peaks = AMPD(rowMean)
        # 提取峰值信息
        for peak in peaks:
            leftBase, rightBase = sleeperBase(rowMean, peak)
            plt.plot(peak, rowMean[peak], "x")

            # 左右轨枕的分数权重为0.4，中间轨枕为0.2
            sleeperRowScore[leftBase:rightBase + 1] += 1

    plt.savefig(os.path.join(segDir, f"row_mean_{timeStr}.svg"), dpi=300, format="svg")
    if show:
        startTime = time.time()
        plt.show()
        endTime = time.time()
        showCostTime += endTime - startTime

    # 提取轨枕点云
    pointColors = np.zeros([points.shape[0], 3])
    pointColors[:] = [0, 255, 0]  # 先为所有点附上绿色
    for mileageIndex in range(mileageNum):
        if sleeperRowScore[mileageIndex] >= 0.6:
            for pIndex in mileageList[mileageIndex]:
                pointColors[pIndex] = [255, 0, 0]  # 轨枕区域用红色标注

    if show:
        startTime = time.time()
        showPointCloud(points, pointColors)
        endTime = time.time()
        showCostTime += endTime - startTime
    return showCostTime


@costTime
def method1d(points, sectionInterval=5e-3, show=False):
    """
    基于一维统计信号的分割方法。
    :param points: (np.array[n, 4]) 输入的点云
    :param sectionInterval: (float) 横截断面内点云的采样间隔（单位：m）
    :param show: (bool) 是否展示结果
    :return: None
    """
    # 先设定计算结果保存目录
    method1dDir = os.path.join(outputDir, "method1d")
    os.makedirs(method1dDir, exist_ok=True)

    showCostTime = 0.0

    # 计算里程信息
    assert points.shape[0] > 0
    mileageList = [[]]
    mileageIndex = 0
    for pointIndex in range(points.shape[0]):
        mileageList[mileageIndex].append(pointIndex)
        if pointIndex < points.shape[0] - 1:
            if points[pointIndex + 1, 0] != points[pointIndex, 0]:
                mileageIndex += 1
                mileageList.append([])

    mileageNum = len(mileageList)
    mileageMeanInterval = abs(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (
            mileageNum - 1)
    if mileageNum > 1:
        logger.info(
            f"里程的排列是否均匀：{(points[mileageList[0][0], 0] - points[mileageList[-1][0], 0]) / (points[mileageList[0][0], 0] - points[mileageList[1][0], 0]) == mileageNum - 1}，"
            f"平均断面间隔为：{mileageMeanInterval}m")
    else:
        logger.error("里程数不足！")
        return showCostTime

    # 生成差分图
    deepMesh, diffMesh, mesh2points, _, _, _, _, _ = generateDiffImage(
        points, mileageList, mileageNum, mileageMeanInterval, -1.2, 1.2,
        sectionInterval, useInterpolation=False)

    colMeanInfo = np.zeros((deepMesh.shape[1],))
    # 逐列像素求均值和标准差
    for col in range(deepMesh.shape[1]):
        colMeanInfo[col] = np.mean(deepMesh[:, col])

    # # 铁轨按列排布，且根据模型图，位于道砟模型图上方；并且，由于其方差小，所以均值较之于别的像素列要高
    # # 由于铁轨有两根，所以只需要找出列均值最大的两个波峰，就算找到了铁轨
    # # height参数限制了波峰的下限，此波峰至少要大于整体均值
    # peaks, peaksInfo = find_peaks(colMeanInfo, height=np.mean(colMeanInfo))
    # # 提取峰值信息
    # peakHeights = peaksInfo['peak_heights']
    #
    # # 先找最大峰
    # peak1Index = np.argmax(peakHeights)
    # peak1 = peaks[peak1Index]
    # # 去掉最大峰，再找次峰
    # peakHeights[peak1Index] = float('-inf')
    # peak2Index = np.argmax(peakHeights)
    # peak2 = peaks[peak2Index]
    #
    # peak1left2, peak1right2 = findPeakBase(colMeanInfo, peak1)
    # peak2left2, peak2right2 = findPeakBase(colMeanInfo, peak2)
    #
    # plt.figure(1)
    # plt.clf()
    # xTicks = np.arange(scanModel[2][0], scanModel[3][0] + sectionInterval, sectionInterval)
    # plt.plot(xTicks, colMeanInfo)
    # plt.plot(peak1 * sectionInterval + scanModel[2][0], colMeanInfo[peak1], "x")
    # plt.plot(peak1left2 * sectionInterval + scanModel[2][0], colMeanInfo[peak1left2], "o")
    # plt.plot(peak1right2 * sectionInterval + scanModel[2][0], colMeanInfo[peak1right2], "o")
    # plt.plot(peak2 * sectionInterval + scanModel[2][0], colMeanInfo[peak2], "x")
    # plt.plot(peak2left2 * sectionInterval + scanModel[2][0], colMeanInfo[peak2left2], "o")
    # plt.plot(peak2right2 * sectionInterval + scanModel[2][0], colMeanInfo[peak2right2], "o")
    # plt.plot(xTicks, np.ones_like(colMeanInfo) * np.mean(colMeanInfo), '--')
    # plt.xlabel('点云横截断面坐标（m）', fontproperties=myFont)
    # plt.ylabel('点云横截断面纵列的高程均值（m）', fontproperties=myFont)
    timeStr = datetime.utcnow().strftime('%H%M%S_%f')[:-3]
    # plt.savefig(os.path.join(method1dDir, f"col_mean_{timeStr}.svg"), dpi=300, format="svg")
    # if show:
    #     startTime = time.time()
    #     plt.show()
    #     endTime = time.time()
    #     showCostTime += endTime - startTime
    #
    # if peak1left2 <= peak2left2:
    #     railRange = [[peak1left2, peak1right2], [peak2left2, peak2right2]]
    # else:
    #     railRange = [[peak2left2, peak2right2], [peak1left2, peak1right2]]
    #
    # 根据找到的范围，提取出分割后的点云
    pointColors = np.zeros([points.shape[0], 3])
    pointColors[:] = [0, 255, 0]  # 先为所有点附上绿色
    # for rail in railRange:
    #     for colIndex in range(rail[0], rail[1]):
    #         for rowIndex in range(deepMesh.shape[0]):
    #             for pIndex in mesh2points[rowIndex][colIndex]:
    #                 pointColors[pIndex] = [0, 0, 255]  # 钢轨部分用蓝色标注

    # if show:
    #     startTime = time.time()
    #     showPointCloud(points, pointColors)
    #     endTime = time.time()
    #     showCostTime += endTime - startTime

    # 计算轨枕范围
    sleeperRange = [[0, round((-0.95 + 1.2) / sectionInterval)],  # 轨枕左侧
                    [round((-0.55 + 1.2) / sectionInterval), round((0.55 + 1.2) / sectionInterval)],  # 轨枕中间
                    [round((0.95 + 1.2) / sectionInterval), deepMesh.shape[1]]]  # 轨枕右侧

    # 然后，逐个对左中右的轨枕区域做均值行投影
    plt.figure(2)
    plt.clf()
    sleeperRowScore = np.zeros([deepMesh.shape[0]])
    for index, sleeperArea in enumerate(sleeperRange):
        rowMeanInfo = np.zeros((deepMesh.shape[0],))
        # 逐行像素求均匀度
        for row in range(deepMesh.shape[0]):
            rowMeanInfo[row] = np.mean(deepMesh[row, sleeperArea[0]:sleeperArea[1]])

        plt.subplot(3, 1, index + 1)
        plt.plot(rowMeanInfo)
        plt.plot(np.ones_like(rowMeanInfo) * np.mean(rowMeanInfo), '--')

        # 提取波峰，峰宽不做约束，但距离两峰之间距离必须大于一个轨枕的断面数
        # peaks, peaksInfo = find_peaks(rowMeanInfo, height=np.mean(rowMeanInfo), width=0,
        #                               distance=sleeperWidth / mileageMeanInterval)
        # AMPD算法
        peaks = AMPD(rowMeanInfo, height=np.mean(rowMeanInfo))
        # 提取峰值信息
        for peak in peaks:
            leftBase, rightBase = findPeakBase(rowMeanInfo, peak)
            plt.plot(peak, rowMeanInfo[peak], "x")

            # 左右轨枕的分数权重为0.4，中间轨枕为0.2
            sleeperRowScore[leftBase:rightBase + 1] += 1

    plt.savefig(os.path.join(method1dDir, f"row_mean_{timeStr}.svg"), dpi=300, format="svg")
    if show:
        startTime = time.time()
        plt.show()
        endTime = time.time()
        showCostTime += endTime - startTime

    # 提取轨枕点云
    for sleeperArea in sleeperRange:
        for colIndex in range(sleeperArea[0], sleeperArea[1]):
            for rowIndex in range(deepMesh.shape[0]):
                # 分数大于等于0.6，也就是两个轨被识别出来
                if sleeperRowScore[rowIndex] >= 0.6:
                    for pIndex in mesh2points[rowIndex][colIndex]:
                        pointColors[pIndex] = [255, 0, 0]  # 轨枕区域用红色标注
    if show:
        startTime = time.time()
        showPointCloud(points, pointColors)
        endTime = time.time()
        showCostTime += endTime - startTime
    return showCostTime


def sleeperBase(signal, peak, yThr=0.02):
    """
    寻找行均值统计信号中每个轨枕波峰对应的范围。
    波基定义为：信号均值附近的点，该点需大于峰值-yThr
    :param signal: (np.array, shape=[n,]) 一维信号
    :param peak: (int) 波峰在一维信号中的位置
    :param yThr: (float) 峰值的下界限
    :return: (tuple(int, int)) 左波谷与右波谷的位置
    """
    leftBase = peak
    rightBase = peak

    # 先往左找
    while leftBase > 0 and signal[leftBase - 1] > signal[peak] - yThr:
        leftBase -= 1

    # 再往右找
    while rightBase < len(signal) - 1 and signal[rightBase + 1] > signal[peak] - yThr:
        rightBase += 1

    return leftBase, rightBase


def findPeakBase(signal, peak, meanThr=0.3, kThr=0.3):
    """
    寻找一维信号中波峰的范围。按先找左波基，再找右波基的顺序进行。
    波基定义为：信号均值附近的点，或者波谷，且该波谷需满足两个条件：
    （1）该波谷应尽量靠近均值，该定义由meanThr约束；
    （2）该波谷到下一个波谷之间的斜率，应该比波谷到波峰的斜率小，该定义由kThr约束。
    :param signal: (np.array, shape=[n,]) 一维信号
    :param peak: (int) 波峰在一维信号中的位置
    :param meanThr: (float) 当(波谷值-均值)/(波峰值-均值)<meanThr时，则确定为左波谷
    :param kThr: (float) 当 次波谷值到波谷值的斜率/波谷值到波峰值的斜率>kThr时，则不为左波谷
    :return: (tuple(int, int)) 左波谷与右波谷的位置
    """
    leftBase = peak
    rightBase = peak

    # 先计算波峰的值和信号均值
    peakValue = signal[peak]
    signalMean = np.mean(signal)

    findLeftBase = False
    while not findLeftBase:
        probableLeftBase = False  # 初始化可能找到的波谷

        # 先找左波谷
        for index in range(leftBase - 1, 0, -1):
            # 如果当前位置值小于均值，则右边的点确定为波基
            if signal[index] <= signalMean:
                leftBase = index + 1
                findLeftBase = True
                break

            # 求当前位置的左导数（左差分）
            difference = signal[index] - signal[index - 1]
            # 当导数小于0时，此时为波谷；进一步的，当前(波谷值-均值)/(波峰值-均值)<meanThr时，则确定为左波谷
            if difference < 0 and (signal[index] - signalMean) / (peakValue - signalMean) < meanThr:
                leftBase = index
                probableLeftBase = True
                break

        # 如果直接确定了波基，退出循环
        if findLeftBase:
            break
        # 假设遍历完左侧都没找到可能的左波谷，则左波谷为0，且防止陷入死循环，直接退出
        elif not probableLeftBase:
            leftBase = 0
            break

        # 再往左侧找一个波峰
        leftPeak = 0
        for index in range(leftBase - 1, 0, -1):
            difference = signal[index] - signal[index - 1]
            if difference > 0:
                leftPeak = index
                break
        
        # 如果没有找到波峰，则对比起点到上一个波谷之间的斜率，与上一个波谷到波峰的斜率，若该斜率绝对值的比值小于meanThr时，则是波谷
        if leftPeak == 0:
            k1 = (signal[leftPeak] - signal[leftBase]) / (leftPeak - leftBase)
            k0 = (signal[leftBase] - peakValue) / (leftBase - peak)
            if k1 / k0 <= kThr:
                # 若是，则最终确定是左波基，退出循环
                findLeftBase = True
            else:
                # 若不是，则把左波峰当成下一次开始的起点
                leftBase = leftPeak
            break

        # 再找下一个波谷
        for index in range(leftPeak - 1, 0, -1):
            # 如果当前位置值小于均值，则右边的点确定为波基
            if signal[index] <= signalMean:
                leftBase = index + 1
                findLeftBase = True
                break

            difference = signal[index] - signal[index - 1]
            if difference < 0:
                # 如果找到了波谷，则对比该波谷到上一个波谷之间的斜率，与上一个波谷到波峰的斜率，若该斜率绝对值的比值小于meanThr时，则是波谷
                k1 = (signal[index] - signal[leftBase]) / (index - leftBase)
                k0 = (signal[leftBase] - peakValue) / (leftBase - peak)
                if k1 / k0 <= kThr:
                    # 若是，则最终确定是左波基，退出循环
                    findLeftBase = True
                else:
                    # 若不是，则把左波峰当成下一次开始的起点
                    leftBase = leftPeak
                break

    findRightBase = False
    while not findRightBase:
        probableRightBase = False  # 初始化可能找到的波谷
        # 再找右波谷
        for index in range(rightBase + 1, signal.shape[0] - 1):
            # 如果当前位置值小于均值，则左边的点确定为波基
            if signal[index] <= signalMean:
                rightBase = index - 1
                findRightBase = True
                break

            # 求当前位置的右导数（右差分）
            difference = signal[index + 1] - signal[index]
            # 当导数大于0时，此时为波谷；进一步的，当前(波谷值-均值)/(波峰值-均值)<meanThr时，则确定为右波谷
            if difference > 0 and (signal[index] - signalMean) / (peakValue - signalMean) < meanThr:
                rightBase = index
                probableRightBase = True
                break

        # 如果直接确定了波基，退出循环
        if findRightBase:
            break
        # 假设遍历完左侧都没找到可能的左波谷，则左波谷为0，且防止陷入死循环，直接退出
        elif not probableRightBase:
            rightBase = signal.shape[0] - 1
            break

        # 再往右侧找一个波峰
        rightPeak = signal.shape[0] - 1
        for index in range(rightBase + 1, signal.shape[0] - 1):
            difference = signal[index + 1] - signal[index]
            if difference < 0:
                rightPeak = index
                break
        
        # 如果没有找到波峰，则对比最终点到上一个波谷之间的斜率，与上一个波谷到波峰的斜率，若该斜率绝对值的比值小于meanThr时，则是波谷
        if rightPeak == signal.shape[0] - 1:
            k1 = (signal[rightPeak] - signal[rightBase]) / (rightPeak - rightBase)
            k0 = (signal[rightBase] - peakValue) / (rightBase - peak)
            if k1 / k0 <= kThr:
                # 若是，则最终确定是右波基，退出循环
                findRightBase = True
            else:
                # 若不是，则最终点是右波谷，退出循环
                rightBase = rightPeak
            break

        # 再找下一个波谷
        for index in range(rightPeak + 1, signal.shape[0] - 1):
            # 如果当前位置值小于均值，则左边的点确定为波基
            if signal[index] <= signalMean:
                rightBase = index - 1
                findRightBase = True
                break

            difference = signal[index + 1] - signal[index]
            if difference > 0:
                # 如果找到了波谷，则对比该波谷到上一个波谷之间的斜率，与上一个波谷到波峰的斜率，若该斜率绝对值的比值小于meanThr时，则是波谷
                k1 = (signal[index] - signal[rightBase]) / (index - rightBase)
                k0 = (signal[rightBase] - peakValue) / (rightBase - peak)
                if k1 / k0 <= kThr:
                    # 若是，则最终确定是右波基，退出循环
                    findRightBase = True
                else:
                    # 若不是，则把右波峰当成下一次开始的起点
                    rightBase = rightPeak
                break

    return leftBase, rightBase


def AMPD(data, height=None):
    """
    实现AMPD算法，参考 https://zhuanlan.zhihu.com/p/549588865
    :param data: 1-D numpy.ndarray
    :param height: 峰值的最低高度
    :return: 波峰所在索引值的列表
    """
    p_data = np.zeros_like(data, dtype=np.int32)
    count = data.shape[0]
    arr_rowsum = []
    for k in range(1, count // 2 + 1):
        row_sum = 0
        for j in range(k, count - k):
            if data[j] > data[j - k] and data[j] > data[j + k]:
                row_sum -= 1
        arr_rowsum.append(row_sum)
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    for k in range(1, max_window_length + 1):
        for j in range(k, count - k):
            if data[j] > data[j - k] and data[j] > data[j + k]:
                p_data[j] += 1
    peaks = np.where(p_data == max_window_length)[0]

    if height is not None:
        tmp = []
        for peak in peaks:
            if data[peak] > height:
                tmp.append(peak)
        peaks = np.array(tmp)

    return np.sort(peaks)


@costTime
def regionGrow(valueImage, segImage):
    """
    区域生长算法。
    :param valueImage: (np.array) 灰度图。
    :param segImage: (np.array) 分割二值图。
    :return: (np.array) 新的分割二值图。
    """
    seedList = []
    pixelValues = []
    nearList = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]  # 8邻域

    # 先提取子连通域的像素坐标和对应原始图的像素值
    for row in range(segImage.shape[0]):
        for col in range(segImage.shape[1]):
            if segImage[row, col] == 255:
                seedList.append([row, col])
                pixelValues.append(valueImage[row, col])

    # 再计算均值和标准差
    pixelValues = np.array(pixelValues)
    mean = np.mean(pixelValues)
    std = np.std(pixelValues)

    # 开始进行区域生长
    while len(seedList) > 0:
        currentSeed = seedList.pop()
        for nearPlus in nearList:
            nearRow = currentSeed[0] + nearPlus[0]
            nearRow = max(0, nearRow)
            nearRow = min(valueImage.shape[0] - 1, nearRow)
            nearCol = currentSeed[1] + nearPlus[1]
            nearCol = max(0, nearCol)
            nearCol = min(valueImage.shape[1] - 1, nearCol)
            nearPixel = [nearRow, nearCol]

            # 首先判断该相邻像素是否已被标记
            if segImage[nearPixel[0], nearPixel[1]] == 255:
                continue
            # 再判断该像素灰度值与均值的绝对差值，是否大于某个阈值
            elif abs(valueImage[nearPixel[0], nearPixel[1]] - mean) > 0.5 * std:
                continue
            # 如果都不是，则该相邻像素可以纳入连通域
            else:
                segImage[nearPixel[0], nearPixel[1]] = 255
                seedList.append(nearPixel)

    return segImage


@costTime
def connectedDomainCompute(inputImage, minArea=300, minX=0, maxX=float("inf"), minIoU=0.5):
    """
    联通域计算，输入二值图，输出二值图。
    :param inputImage: (np.array) 需要处理的二值图。
    :param minArea: (int)联通域的最小面积（像素数量）
    :param minX: (float)联通域所在中心点应大于该值。
    :param maxX: (float)联通域所在中心点应小于该值。
    :param minIoU: (float)联通域面积/联通域最小外接矩形的比值应大于该值。
    :return: (tuple(np.array, list[np.array]) 计算好的联通域的总二值图，以及对应的每个子联通域的二值图。
    """
    numLabels, labels, states, centroids = cv2.connectedComponentsWithStats(inputImage, connectivity=8)
    usedLabels = []
    for index in range(1, numLabels):
        area = states[index, 4]  # 面积
        w = states[index, 2]  # 宽
        h = states[index, 3]  # 高
        centerX = centroids[index, 0]  # 中心点的x坐标
        if area < minArea:
            continue
        elif centerX < minX or centerX > maxX:
            continue
        elif area / (w * h) < minIoU:
            continue
        else:
            usedLabels.append(index)
    # 遍历像素，获得二值图
    subSegImages = [np.zeros((inputImage.shape[0], inputImage.shape[1]), np.uint8) for _ in range(len(usedLabels))]
    wholeSegImage = np.zeros((inputImage.shape[0], inputImage.shape[1]), np.uint8)
    for row in range(inputImage.shape[0]):
        for col in range(inputImage.shape[1]):
            for index, label in enumerate(usedLabels):
                if labels[row, col] == label:
                    wholeSegImage[row, col] = 255
                    subSegImages[index][row, col] = 255
    return wholeSegImage, subSegImages


def modelY(pointZ):
    """
    计算对应z值点的模型高度model y
    :param pointZ: (float) 点的z值，单位：m
    :return: (float) 点的模型高度，单位：m
    """
    if pointZ < scanModel[0][0] or pointZ > scanModel[-1][0]:
        return 0.0
    else:
        for index, modelPara in enumerate(scanModel):
            if scanModel[index][0] <= pointZ <= scanModel[index + 1][0]:
                leftZ, leftY = scanModel[index]
                rightZ, rightY = scanModel[index + 1]
                k = (rightY - leftY) / (rightZ - leftZ)
                diffZ = pointZ - leftZ
                return leftY + k * diffZ


def diffMap(pixelX, zMin, pixelInterval):
    """
    计算差分图对应像素所代表的模板点的x坐标(对应像素的y坐标)和z坐标（对应像素的x坐标）
    :param pixelX: (float)  像素点的x坐标
    :param zMin: (float) 断面的z值下界
    :param pixelInterval: (float) 横截断面内点云采样间隔（单位：m）
    :return: (float) 对应模板点的z坐标
    """
    pointZ = zMin + pixelX * pixelInterval
    return pointZ


@costTime
def generateDiffImage(points, mileageList, mileageNum, mileageMeanInterval, zMin, zMax, pixelInterval,
                      useInterpolation=True):
    """
    生成以模板为基准的差分图和归一化的灰度图
    :param points: (np.array[n, 4]) 输入的点云
    :param mileageList: (list) 按横截断面储存点云的列表
    :param mileageNum: (int) 横截断面的数量
    :param mileageMeanInterval: (float) 横截断面之间的平均间隔
    :param zMin: (float) 网格化的z值下界
    :param zMax: (float) 网格化的z值上界
    :param pixelInterval: (float) 横截断面内点的采样间隔（单位：m）
    :param useInterpolation: (bool)  是否开启插值，默认开启
    :return: tuple((np.array, 高程值投影网格), (np.array, 差分投影网格), (list, 网格索引对应的点云集合),
             (np.array, 归一化差分图), (np.array, 绝对值归一化差分图), (np.array, 深度图), (list, 图像索引对应的点云集合))
    """
    railOutside = 0.95
    railInside = 0.55

    colNum = round((zMax - zMin) / pixelInterval) + 1
    rowNum = mileageNum
    diffMesh = np.zeros([rowNum, colNum])
    deepMesh = np.zeros([rowNum, colNum])
    mesh2points = [[] for _ in range(rowNum)]  # 用来保存网格点到点云的映射

    for rowIndex in range(rowNum):
        colIndex = 0  # 当前填充的差分图的列索引
        mesh2points[rowIndex] = [[] for _ in range(colNum)]  # 当前行的列索引囊括的实际点云列表
        sectionList = mileageList[rowIndex]
        sectionNum = len(sectionList)
        sectionIndex = 0
        while sectionIndex < sectionNum:
            idealPointZ = diffMap(colIndex, zMin, pixelInterval)
            pointZ = points[sectionList[sectionIndex], 2]
            pointY = points[sectionList[sectionIndex], 1]

            # 由于插值需要下一个点，因此若没有下一个点可插值，则就用断面最后一个点的信息补全
            if sectionIndex == sectionNum - 1:
                diffMesh[rowIndex, colIndex] = pointY - modelY(pointZ)
                deepMesh[rowIndex, colIndex] = pointY
                # 遍历完一行了就退出while循环
                if colIndex == colNum - 1:
                    break
                else:
                    colIndex += 1
                    mesh2points[rowIndex].append([])
            else:
                nextPointZ = points[sectionList[sectionIndex + 1], 2]
                nextPointY = points[sectionList[sectionIndex + 1], 1]
                # 若目标点的坐标在当前点的左侧，则继续遍历下去也达不到插值条件
                if idealPointZ < pointZ:
                    diffMesh[rowIndex, colIndex] = pointY - modelY(pointZ)
                    deepMesh[rowIndex, colIndex] = pointY
                    if colIndex == colNum - 1:
                        break
                    else:
                        colIndex += 1
                        mesh2points[rowIndex].append([])
                elif pointZ <= idealPointZ <= nextPointZ:
                    if pointZ == nextPointZ:
                        idealPointY = (nextPointY + pointY) / 2
                    else:
                        k = (nextPointY - pointY) / (nextPointZ - pointZ)
                        idealPointY = pointY + k * (idealPointZ - pointZ)

                    diffMesh[rowIndex, colIndex] = idealPointY - modelY(idealPointZ)
                    deepMesh[rowIndex, colIndex] = idealPointY
                    if colIndex == colNum - 1:
                        break
                    else:
                        colIndex += 1
                        mesh2points[rowIndex].append([])
                        # 无法插值又不是异常点，则遍历下一个断面的点
                else:
                    # 在要遍历下一个断面点之前，将该点添加到所在网格交点的点云集中
                    nearColIndex = round((pointZ - zMin) / pixelInterval)
                    mesh2points[rowIndex][nearColIndex].append(sectionList[sectionIndex])
                    sectionIndex += 1

    # 再根据平均断面间隔和断面内点的间隔，设置合适的断面插值数
    interpolationTimes = int(np.floor(mileageMeanInterval / pixelInterval))  # 插值倍数
    if useInterpolation and interpolationTimes > 1:
        image2points = []  # 图像像素到点云集的映射
        newRowNum = (rowNum - 1) * interpolationTimes + 1
        deepImage = np.zeros([newRowNum, colNum])
        diffImage = np.zeros([newRowNum, colNum])
        # 根据插值数，逐断面进行插值操作
        for newRowIndex in range(newRowNum):
            leftOldRow = int(np.floor(newRowIndex / interpolationTimes))  # 被插值的左断面
            rightOldRow = min(leftOldRow + 1, rowNum - 1)  # 被插值的右断面
            # 深度图插值
            deepImage[newRowIndex, :] = \
                deepMesh[leftOldRow, :] + \
                (deepMesh[rightOldRow, :] - deepMesh[leftOldRow, :]) / interpolationTimes * \
                (newRowIndex % interpolationTimes)
            # 差分图插值
            diffImage[newRowIndex, :] = \
                diffMesh[leftOldRow, :] + \
                (diffMesh[rightOldRow, :] - diffMesh[leftOldRow, :]) / interpolationTimes * \
                (newRowIndex % interpolationTimes)
            # 像素到点云集的映射复制，就近原则处理，用余数来判断
            if newRowIndex % interpolationTimes < interpolationTimes / 2:
                image2points.append(copy.deepcopy(mesh2points[leftOldRow]))
            else:
                image2points.append(copy.deepcopy(mesh2points[rightOldRow]))

        absImage = diffImage.copy()
        absImage = np.abs(absImage)
        absImage = cv2.normalize(absImage, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        normImage = diffImage.copy()
        normImage = cv2.normalize(normImage, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        deepImage = cv2.normalize(deepImage, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    else:
        image2points = copy.deepcopy(mesh2points)

        absImage = diffMesh.copy()
        absImage = np.abs(absImage)
        absImage = cv2.normalize(absImage, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        normImage = diffMesh.copy()
        normImage = cv2.normalize(normImage, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        deepImage = cv2.normalize(deepMesh, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return deepMesh, diffMesh, mesh2points, normImage, absImage, deepImage, image2points


def showPointCloud(inputPoints, colors=None, save=True, show = True):
    """
    使用open3d进行点云可视化。
    :param inputPoints: (np.array) 输入的点云
    :param colors: (np.array) 输入的点云对应的颜色，形状与inputPoints一致，通道顺序为RGB
    :return: None
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(inputPoints[:, :3])
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if show:
        o3d.visualization.draw_geometries([pcd])
    if save:
        outputDir = "E:/lcy/dgcnn/location_output"
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        outputDir = os.path.join(outputDir, f"{slideIndex}.pcd")
        o3d.io.write_point_cloud(outputDir,pcd)
        print(f"Point cloud segment {slideIndex} saved to {outputDir}")
def save_diffmesh_image(diffMesh, timeStr, outputDir):
    """
    将 diffMesh 保存为图像文件。
    
    :param diffMesh: (np.array) 二维数组，表示差分图像
    :param timeStr: (str) 当前时间字符串，用作文件名的一部分
    :param outputDir: (str) 输出目录
    :return: None
    """
    # 创建保存目录
    method1dDir = os.path.join(outputDir, "method1d")
    os.makedirs(method1dDir, exist_ok=True)
    
    # 使用imshow显示diffMesh，cmap='gray'表示灰度图
    plt.imshow(diffMesh, cmap='gray', aspect='equal')  # 设置aspect为'equal'以保持图像比例
    plt.colorbar()  # 添加颜色条，帮助标识灰度值
    
    # 不显示横坐标和纵坐标
    # plt.xticks([])  # 清除x轴刻度
    # plt.yticks([])  # 清除y轴刻度
    
    # 不显示横坐标和纵坐标的标题
    # plt.xlabel('')  # 清除x轴标题
    # plt.ylabel('')  # 清除y轴标题
    
    # 移除图像周围的边框
    # plt.axis('off')
    
    # 保存图像为SVG格式
    plt.savefig(os.path.join(method1dDir, f"diffMesh_{timeStr}.svg"), dpi=300, format="svg", bbox_inches='tight', pad_inches=0)
    
    # 可选：也可以保存为其他格式，如PNG
    plt.savefig(os.path.join(method1dDir, f"diffMesh_{timeStr}.png"), dpi=300, format="png", bbox_inches='tight', pad_inches=0)
    
    # 清理图像缓存
    plt.clf()

# 模板单位与点云单位缩放比例
scaleRatio = 1e-3
# 轨枕宽度（单位m）
sleeperWidth = 0.3
# 模板，注意需要按照z值（也就是剖面图的x坐标）从小到大的顺序填写
scanModel = [[-2412.5, -614], [-1537.5, -114], [-1300, -264], [1300, -264], [1537.5, -114], [2412.5, -614]]
for i, para in enumerate(scanModel):
    scanModel[i][0] *= scaleRatio
    scanModel[i][1] *= scaleRatio

slideIndex = 0
SlideCount = -1
sleeperMiddleZ = []
bridgeLeftZ = []
bridgeRightZ = []
bridgeMileageList = []
if __name__ == "__main__":
    myFont = font_manager.FontProperties(fname="C:/WINDOWS/Fonts/STSONG.TTF") # 设置字体属性

    currentFileDir = os.path.dirname(os.path.abspath(__file__)) # 返回当前执行脚本的绝对路径的目录部分
    outputDir = os.path.join(currentFileDir, "output")
    outputDir = os.path.join(outputDir, time.strftime("%Y_%m_%d_%H_%M", time.localtime()))
    logger.add(os.path.join(outputDir, 'log.txt'))

    # 读取las文件
    # lasFilePath = "F:/zxr/项目数据/24-1-11/成渝线/成渝线_上行_普通线路_2019-11-09_7-48-19_0.las"
    lasFilePath = "E:/lcy/项目数据/大理/split_x_1624_2011.las"
    # lasFilePath = 'E:/zxr/项目数据/道床/现场数据/盈/西安/宝成线_下行_普通线路_2022-03-05_15-28-23 - 1_0.las'
    # lasFilePath = 'F:/zxr/项目数据/道床/现场数据/欠/沈阳/新通线_上行_普通线路_2019-03-09_23-36-5_0.las'
    # lasFilePath = 'test.las'

    # lasFile = las.read(lasFilePath)
    # pointsNum = lasFile.header.point_count
    # logger.info(f"读取las文件{lasFilePath}，共读取到{pointsNum}个点...")
    #
    # wholePoints = lasFile.xyz
    # intensity = lasFile.intensity
    # pointId = lasFile.point_source_id
    # wholePoints = np.concatenate([wholePoints, intensity.reshape([wholePoints.shape[0], 1]), pointId.reshape([wholePoints.shape[0], 1])], axis=1)
    #
    # 存一下文件，不然文件太大了
    # np.save(os.path.join(os.path.dirname(lasFilePath), 'guardrail1.npy'), wholePoints[80000000:100000000, :])
    # np.save(os.path.join(os.path.dirname(lasFilePath), 'tunnel.npy'), wholePoints[126000000:142000000, :])
    # np.save(os.path.join(os.path.dirname(lasFilePath), 'guardrail2.npy'), wholePoints[154000000:228000000, :])
    # np.save(os.path.join(os.path.dirname(lasFilePath), 'switch1.npy'), wholePoints[268000000:278000000, :])
    # np.save(os.path.join(os.path.dirname(lasFilePath), 'switch2.npy'), wholePoints[283000000:293000000, :])
    # np.save(os.path.join(os.path.dirname(lasFilePath), 'platform.npy'), wholePoints[341000000:355000000, :])
    #inputDir = "E:/lcy/dgcnn/data/coordinate_correction/bridge2_las/"
    # 获取目录下的所有.las文件
    #lasFiles = [f for f in os.listdir(inputDir) if f.endswith('.las')]

    #for lasFile in lasFiles:
        #lasFilePath = os.path.join(inputDir, lasFile)
    logger.info(f"正在处理文件: {lasFilePath}")
    lasFile = las.read(lasFilePath)
    wholePoints = lasFile.xyz
    intensity = lasFile.intensity
    pointId = lasFile.point_source_id
    wholePoints = np.concatenate([wholePoints, intensity.reshape([wholePoints.shape[0], 1]), pointId.reshape([wholePoints.shape[0], 1])], axis=1)
    np.save(os.path.join(os.path.dirname(lasFilePath), 'tmp.npy'), wholePoints)
    # 读取临时文件
    wholePoints = np.load(os.path.join(os.path.dirname(lasFilePath), 'tmp.npy'))
    pointsNum = wholePoints.shape[0]
    
    pointColors = np.zeros([wholePoints.shape[0], 3])
    startPointsIndex = 0
    clipPointsNum = 1000000
    # clipPointsNum = pointsNum
    logger.info(f"开始以{clipPointsNum}个点为切片，遍历点云...")
    run_time = 0
    while startPointsIndex < pointsNum:
        slideIndex += 1
        SlideCount += 1
        sleeperMiddleZ.append([])
        bridgeLeftZ.append([])
        bridgeRightZ.append([])
        bridgeMileageList.append([])
        endPointsIndex = min(startPointsIndex + clipPointsNum, pointsNum)
        clipPoints = wholePoints[startPointsIndex: endPointsIndex, :] # clipPoints包含startPointsIndex至endPointsIndex-1行的数据

        # 先按x排序，再按z排序，最后按y排序
        sortIndex = np.lexsort((clipPoints[:, 1], clipPoints[:, 2], clipPoints[:, 0]))
        clipPoints = clipPoints[sortIndex, :]

        # 点云可视化
        # if showFig:
        #     showPointCloud(clipPoints)

        # 按z值（横向）进行点云的过滤
        railwayPoints = clipPoints[np.where(clipPoints[:, 2] >= -3.7)]
        railwayPoints = railwayPoints[np.where(railwayPoints[:, 2] <= 3.7)]

        # 按y值（高度）进行点云过滤
        railwayPoints = railwayPoints[np.where(railwayPoints[:, 1] <= 0.2)]
        railwayPoints = railwayPoints[np.where(railwayPoints[:, 1] >= -0.9)]

        #showPointCloud(railwayPoints)

        # 使用神经网络
        # seg2d(railwayPoints, show=showFig)
        # seg3d(railwayPoints, show=showFig)

        # 开题报告方法一
        # method1d(railwayPoints, show=showFig)

        # 开题报告方法二
        # method2d(railwayPoints, show=showFig)

        # 新的横截扫面断面分割法
        bridgeCal(railwayPoints, show=False)
        #windowSlide(railwayPoints, show=True)
        # 新的统计分割法
        # statisticalSeg(railwayPoints, show=showFig)

        # 异物去噪！
        # errorSeg2(railwayPoints, show=showFig)

        # 护轨、站台分割
    #     start_time = time.time()
    #     railSeg(railwayPoints, show=False)
    #     run_time += time.time() - start_time
    #     if len(bridgeMileageList[-1]) == 0:
    #         bridgeMileageList.pop()
    #     elif len(bridgeMileageList) >= 2:
    #         if(abs(bridgeMileageList[-1][0] - bridgeMileageList[-2][-1]) <= 0.15):
    #             bridgeMileageList[-2][-1] = bridgeMileageList[-1][-1]
    #             bridgeMileageList.pop()
    #     # if showFig:
    #     #     showPointCloud(railwayPoints)

        startPointsIndex = endPointsIndex
        
    # pointColors[:] = [0, 255, 0]  # 先为所有点附上绿色
    # for mileage_range in bridgeMileageList:
    #     start_mileage, end_mileage = mileage_range

    #     # 根据起始里程和结束里程，选择对应范围内的点
    #     mask = (wholePoints[:, 0] >= start_mileage) & (wholePoints[:, 0] <= end_mileage)

    #     # 更新符合条件的点的颜色为蓝色 [0, 0, 255]
    #     pointColors[mask] = [0, 0, 255]  # 蓝色
    #     logger.info(f"桥梁位置为{start_mileage} 至 {end_mileage}")
    #     logger.info(f"总耗时为{run_time}")
    #     showPointCloud(wholePoints, pointColors, show=True)
        if len(bridgeLeftZ[SlideCount]) == 0 or len(bridgeRightZ[SlideCount]) == 0:
            SlideCount -= 1
            sleeperMiddleZ.pop()
            bridgeLeftZ.pop()
            bridgeRightZ.pop()
    bridgeEccentricLen = bridgeEccentricLen / mileageCount
    logger.info(f"桥梁偏心距离为{bridgeEccentricLen *1000}")
