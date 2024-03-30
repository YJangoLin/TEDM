# -*- coding: utf-8 -*-
# @Time    : 2024/1/11 8:16
# @Author  : ZL.Liang
# @FileName: VocToYolo.py
# @Software: PyCharm
# @Blog    ：https://github.com/YJangoLin
import math
import os

import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linear_sum_assignment


# point1:(n, 2) point(n, 2)
def get_cost(rectPoints, polyPoints):
    assert rectPoints.shape == polyPoints.shape, print(rectPoints, polyPoints)
    n = rectPoints.shape[0]
    cost = np.zeros((n, n), dtype=float)
    for ind1, p1 in enumerate(rectPoints):
        cls1 = int(p1[0])
        point1 = p1[1:]
        for ind2, p2 in enumerate(polyPoints):
            cls2 = int(p2[0])
            point2 = p2[1:]
            if cls1 != cls2:
                cost[ind1][ind2] = 9999999999999
                # cost[ind][ind1] = cost[ind1][ind]
                continue
            cost[ind1][ind2] = math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1],
                                                                             2)  # 行代表rect，列代表poly
    col_ind = np.argmin(cost, axis=1)
    return col_ind


def hungarian(cost):
    row_ind, col_ind = linear_sum_assignment(-cost)
    return row_ind, col_ind


def get_center(filePath):
    with open(filePath, 'r') as f:
        return np.array([con.replace('\n', '').split(' ')[:3] for con in f.readlines()], dtype=float)


# 不区分类别
def VocToYolo(vocFilePath, yoloOutputDir, isRect=False):
    tree = ET.parse(vocFilePath)  # 解析xml文件
    root = tree.getroot()
    sizeTag = root.find('size')
    height = int(sizeTag.find('height').text)
    width = int(sizeTag.find('width').text)
    objectsTag = root.findall('object')
    sep = ' '
    fileName = vocFilePath.split('/')[-1].split('.')[0] + '.txt'
    filePath = yoloOutputDir + '/' + fileName
    f = open(filePath, 'a')
    for objectTag in objectsTag:
        className = objectTag.find('name').text
        bndboxTag = objectTag.find('bndbox')
        if isRect:
            if className == 'circle':
                classId = 0
            else:
                classId = 1
            xmin = float(bndboxTag.find('xmin').text)
            ymin = float(bndboxTag.find('ymin').text)
            xmax = float(bndboxTag.find('xmax').text)
            ymax = float(bndboxTag.find('ymax').text)
            # 归一化 --> 转换相对为中心点+长宽
            # 中心点：
            centerx, centery = (xmin + xmax) / 2, (ymin + ymax) / 2
            rl_centerx, rl_centery = centerx / width, centery / height
            # 相对高度
            rl_width, rl_he = abs((xmax - xmin)) / width, abs((ymax - ymin) / height)
            string = str(classId) + sep + str(rl_centerx) + sep + str(rl_centery) + sep + str(rl_width) + sep + str(
                rl_he)
        else:
            # 长宽
            centerx, centery = eval(bndboxTag.find('center').text)  # 获取中心坐标
            centerx, centery = float(centerx), float(centery)  # 范围：[0， 1]
            if className == 'circle':
                classId = 0
                # 获取半径
                R = bndboxTag.find('R').text
                # 形状必须均匀
                string = str(classId) + sep + str(centerx) + sep + str(centery) + sep + str(R) + sep + str(
                    R) + sep + '0'
            else:
                classId = 1
                a, b = eval(bndboxTag.find('axis').text)  # 无法进行相对距离
                angle = float(bndboxTag.find('angle').text)  # 相对角度
                string = str(classId) + sep + str(centerx) + sep + str(centery) + sep + str(a) + sep + str(
                    b) + sep + str(
                    angle)
        f.write(string + '\n')
    f.close()


def match(rectTxtFile, polyTxtFile, w=1024, h=768):
    rectPoints = get_center(rectTxtFile)
    rectPoints[:, 1] = rectPoints[:, 1] * w
    rectPoints[:, 2] = rectPoints[:, 2] * h
    polyPoints = get_center(polyTxtFile)
    col_ind = get_cost(rectPoints, polyPoints)
    print(f'match indexes {col_ind}')
    if len(np.unique(col_ind)) == len(col_ind):
        return col_ind
    else:
        print(f'{polyTxtFile}文件标记错误')
        return None


from PIL import Image, ImageDraw


def draw_image(old_points, amd_points=None, imagePath='../data/images/train1.png'):
    # colors = ['red', 'white', 'black', 'orange']
    old_point = old_points[:, 1:].tolist()
    if amd_points is not None:
        image = Image.open(imagePath)
        draw = ImageDraw.Draw(image)
        amd_point = amd_points[:, 1:].tolist()
        for ind in range(len(amd_point)):
            # draw.text(old_point[ind], str(ind), fill='red')
            draw.text(amd_point[ind], str(ind), fill='white')  # poly
            # draw.point(old_point[ind], fill='red')
            draw.point(amd_point[ind], fill='white')
        image.save('rectLabels.jpg')
        image.show()
    image = Image.open(imagePath)
    draw = ImageDraw.Draw(image)
    for ind1 in range(len(old_point)):
        draw.text(old_point[ind1], str(ind1), fill='blue')  # rect
        draw.point(old_point[ind1], fill='blue')
    image.save('polyLabels.jpg')
    image.show()


def writeTxt(filePath, indexs):
    with open(filePath, 'r') as f:
        context = [con for con in f.readlines()]
    with open(filePath, 'w') as f:
        for ind in indexs:
            f.write(context[ind])


def buildTxt(inputXmlDir, outputTxtDir):
    vocFilePaths = [inputXmlDir + '/' + 'rectLabel', inputXmlDir + '/' + 'polyLabel']  # 先rect再poly
    yoloOutputDir = [outputTxtDir + '/' + 'rectLabel', outputTxtDir + '/' + 'polyLabel']
    isRect = [True, False]
    print('start to switch file to the format of yolo....')
    for ind, vocFilePath in enumerate(vocFilePaths):
        fileList = os.listdir(vocFilePaths[ind])
        if not os.path.exists(yoloOutputDir[ind]): os.makedirs(yoloOutputDir[ind])
        for file in fileList:
            VocToYolo(vocFilePath + '/' + file, yoloOutputDir[ind], isRect=isRect[ind])
    print('The file format conversion is complete!')
    print('Matching in progress ...')

    fileList = os.listdir(yoloOutputDir[0])
    for file in fileList:
        print(f'load {file}')
        indexes = match(yoloOutputDir[0] + '/' + file, yoloOutputDir[1] + '/' + file)
        if indexes is None:
            continue
        writeTxt(yoloOutputDir[1] + '/' + file, indexes)

# if __name__ == '__main__':
#     vocFilePaths = ['./data/xmlData/rectLabel', './data/xmlData/polyLabel']  # 先rect再poly
#     yoloOutputDir = ['./data/rectLabels', './data/polyLabels']
#     isRect = [True, False]
#     for ind, vocFilePath in enumerate(vocFilePaths):
#         fileList = os.listdir(vocFilePaths[ind])
#         if not os.path.exists(yoloOutputDir[ind]): os.makedirs(yoloOutputDir[ind])
#         for file in fileList:
#             VocToYolo(vocFilePath + '/' + file, yoloOutputDir[ind], isRect=isRect[ind])
#     # 重新调整poly的顺序
#     rectTxtPath = './data/rectLabels'
#     polyTxtPath = './data/polyLabels'
#     fileList = os.listdir(rectTxtPath)
#     for file in fileList:
#         print(f'load {file}')
#         indexes = match(rectTxtPath + '/' + file, polyTxtPath + '/' + file)
#         if indexes is None:
#             continue
#         # print(f'match indexes {indexes}')
#         writeTxt(polyTxtPath + '/' + file, indexes)

# 验证匹配是否成功
# imagePath = '../data/images/train102.png'
# rectPath = './data/rectLabels/train102.txt'
# polyPath = './data/polyLabels/train102.txt'
#
# # draw_image(rectPoints, polyPoints, imagePath)
# print(match(rectPath, polyPath))

# rect_points = get_center('./data/rectLabels/train8.txt')
# rect_points[:, 1] = rect_points[:, 1]*1024
# rect_points[:, 2] = rect_points[:, 2] * 768
# draw_image(old_points=rect_points, imagePath='./data/images/train8.png')
