# -*- coding: utf-8 -*-
# @Time    : 2023/12/18 18:52
# @Author  : ZL.Liang
# @FileName: DataSwitch.py
# @Software: PyCharm
# @Blog    ：https://github.com/YJangoLin
import math
from PIL import Image
from Amendment import get_info

# labelDir = './data/labelmeData'
# xmlPath = './data/xmlData'

import os
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import curve_fit


def getLabelFilePath(labelDir):
    return os.listdir(labelDir)


def getR(points):
    centerx, centery = points[0][0], points[0][1]
    tempx, tempy = points[1][0], points[1][1]
    return math.sqrt(math.pow(centerx - tempx, 2) + math.pow(centery - tempy, 2))


def showTags(tag):
    for elem in tag.iter():
        print(elem.tag, elem.text)


# 计算长轴和短轴长
# def fitEllipseFunc(data, x0, y0):
#     k =
# todo:矩形跟圆应该分开存储,先存储圆标记的
def addObject(parentNode, shape, is_rect=False):
    objectTag = ET.SubElement(parentNode, 'object')
    # objectTag = parentNode.find('object')
    ET.SubElement(objectTag, 'name')
    ET.SubElement(objectTag, 'bndbox')
    # # todo bndbox为None？？？
    # showTags(parentNode)
    bndbox = objectTag.find('bndbox')
    if is_rect:
        ET.SubElement(bndbox, 'xmin')
        ET.SubElement(bndbox, 'ymin')
        ET.SubElement(bndbox, 'xmax')
        ET.SubElement(bndbox, 'ymax')
    else:
        if shape == 'polygon':
            ET.SubElement(bndbox, 'center')
            ET.SubElement(bndbox, 'points')
            ET.SubElement(bndbox, 'axis_func')
            ET.SubElement(bndbox, 'fit_points')
            ET.SubElement(bndbox, 'axis')
            ET.SubElement(bndbox, 'angle')
        if shape == 'circle':
            ET.SubElement(bndbox, 'points')
            ET.SubElement(bndbox, 'center')
            ET.SubElement(bndbox, 'R')
    return parentNode


# 字典列表等为可变参数可以进行修改，但Element无法再实参中对函数进行修改
def addXmlNode(parent, Tag):
    ET.SubElement(parent, Tag)
    return parent


def updateItem(ParentTag, Tag, text):
    tag = ParentTag.find(Tag)
    tag.text = str(text)


def writeXml(root, filePath):
    tree = ET.ElementTree(root)
    tree.write(filePath, encoding='utf-8', xml_declaration=True)
    print(f'write to {filePath}')


def getXmlHead(folder_text, imagePath):
    image = Image.open(imagePath)
    root = ET.Element("annotation")
    ET.SubElement(root, 'folder')
    ET.SubElement(root, 'filename')
    ET.SubElement(root, 'size')
    sizeTag = root.find('size')
    ET.SubElement(sizeTag, 'width')
    ET.SubElement(sizeTag, 'height')
    ET.SubElement(sizeTag, 'depth')
    updateItem(root, 'folder', folder_text)
    updateItem(root, 'filename', imagePath.split('/')[-1])
    updateItem(sizeTag, 'width', image.size[0])
    updateItem(sizeTag, 'height', image.size[1])
    updateItem(sizeTag, 'depth', 3)
    return root


# 将json转换为xml
def updatePolygonDataXML(jsonData, imagePath, ouputDir):
    rootXml = getXmlHead('polygonData', imagePath)
    shapesJson = jsonData.get('shapes')
    for i, shape in enumerate(shapesJson):
        # print(f"================{i}====================")
        shape_type = shape.get('shape_type')
        # print(f'shape_type:{shape_type}')
        if shape_type == 'polygon':
            rootXml = addObject(rootXml, shape_type, False)
            objectTag = rootXml.findall('object')[-1]
            updateItem(objectTag, 'name', 'ellipse')
            points = shape.get('points')
            dict = get_info(points)
            bndboxTag = objectTag.find('bndbox')
            updateItem(bndboxTag, 'points', points)
            for key in dict.keys():
                # print(key, dict.get(key))
                updateItem(bndboxTag, key, dict.get(key))
            # print('=================objectTag==================')
            # showTags(objectTag)
        if shape_type == 'circle':
            rootXml = addObject(rootXml, shape_type, False)
            objectTag = rootXml.findall('object')[-1]
            updateItem(objectTag, 'name', 'circle')
            points = shape.get('points')
            bndboxTag = objectTag.find('bndbox')
            updateItem(bndboxTag, 'points', points)
            updateItem(bndboxTag, 'center', points[0])
            updateItem(bndboxTag, 'R', getR(points))
            # print('=================objectTag==================')
            # showTags(objectTag)
    filePath = ouputDir + '/' + imagePath.split('/')[-1].split('.')[0] + '.xml'
    writeXml(rootXml, filePath)


def updateRectDataXML(jsonData, imagePath, ouputDir):
    rootXml = getXmlHead('RectData', imagePath)
    # showTags(rootXml)
    shapesJson = jsonData.get('shapes')
    for i, shape in enumerate(shapesJson):
        # print(f"================{i}====================")
        shape_type = shape.get('shape_type')
        if shape_type == 'rectangle':
            rootXml = addObject(rootXml, shape_type, True)
            objectTag = rootXml.findall('object')[-1]
            updateItem(objectTag, 'name', 'ellipse')
            points = shape.get('points')
            bndboxTag = objectTag.find('bndbox')
            updateItem(bndboxTag, 'xmin', points[0][0])
            updateItem(bndboxTag, 'ymin', points[0][1])
            updateItem(bndboxTag, 'xmax', points[1][0])
            updateItem(bndboxTag, 'ymax', points[1][1])
        if shape_type == 'circle':
            rootXml = addObject(rootXml, shape_type, True)
            objectTag = rootXml.findall('object')[-1]
            updateItem(objectTag, 'name', 'circle')
            points = shape.get('points')
            r = getR(points)
            bndboxTag = objectTag.find('bndbox')
            centerPoint = points[0]
            updateItem(bndboxTag, 'xmin', centerPoint[0] - r)
            updateItem(bndboxTag, 'ymin', centerPoint[1] - r)
            updateItem(bndboxTag, 'xmax', centerPoint[0] + r)
            updateItem(bndboxTag, 'ymax', centerPoint[1] + r)
    filePath = ouputDir + '/' + imagePath.split('/')[-1].split('.')[0] + '.xml'
    writeXml(rootXml, filePath)


def read_json(filePath):
    with open(filePath, 'r+', encoding='utf-8') as f:
        jsonData = json.load(f)
        return jsonData


# def main(labelmeFilePath, isRect=False):
#     jsonData = read_json(labelmeFilePath)
#     if isRect:

def buildXml(imageDir, jsonLabelDir, outputXml):
    outputPolyDir = outputXml + '/' + 'polyLabel'
    outputRectDir = outputXml + '/' + 'rectLabel'
    if not os.path.exists(outputPolyDir): os.makedirs(outputPolyDir)
    if not os.path.exists(outputRectDir): os.makedirs(outputRectDir)
    # jsonData = read_json(labelmeFilePath)
    filesList = getLabelFilePath(labelDir=jsonLabelDir)
    print(f"the number of files: {str(len(filesList))}")
    for i, fileName in enumerate(filesList):
        print(f"================={i + 1}/{len(filesList)}===========================")
        print(f"start to switch {fileName}")
        filePath = jsonLabelDir + '/' + fileName
        imagePath = imageDir + '/' + fileName.split(".")[0] + '.png'
        jsonData = read_json(filePath)
        updatePolygonDataXML(jsonData=jsonData, imagePath=imagePath, ouputDir=outputPolyDir)
        updateRectDataXML(jsonData=jsonData, imagePath=imagePath, ouputDir=outputRectDir)

# if __name__ == '__main__':
#     labelmeDir = 'data/jsonLabels'
#     imageDir = 'data/images'
#     # templateXmlPath = './data/template.xml'
#     outputPolyDir = './data/xmlData/polyLabel'
#     outputRectDir = './data/xmlData/rectLabel'
#     if not os.path.exists(outputPolyDir): os.makedirs(outputPolyDir)
#     if not os.path.exists(outputRectDir): os.makedirs(outputRectDir)
#     # jsonData = read_json(labelmeFilePath)
#     filesList = getLabelFilePath(labelDir=labelmeDir)
#     print(f"the number of files: {str(len(filesList))}")
#     for i, fileName in enumerate(filesList):
#         print(f"================={i + 1}/{len(filesList)}===========================")
#         print(f"start to {fileName}")
#         filePath = labelmeDir + '/' + fileName
#         imagePath = imageDir + '/' + fileName.split(".")[0] + '.png'
#         jsonData = read_json(filePath)
#         updatePolygonDataXML(jsonData=jsonData, imagePath=imagePath, ouputDir=outputPolyDir)
#         updateRectDataXML(jsonData=jsonData, imagePath=imagePath, ouputDir=outputRectDir)
