# -*- coding: utf-8 -*-
# @Time    : 2024/1/14 19:00
# @Author  : ZL.Liang
# @FileName: dataset.py
# @Software: PyCharm
# @Blog    ：https://github.com/YJangoLin

import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.util import getBoxInfo, polyLabels_preDeal, getPolyLabels
import torchvision as tv
import cv2

class MEADataset(Dataset):
    def __init__(self, imagePath: list,roi_size=128, t=0):
        self.imagesPath = self.getImagePath(imagePath)
        self.polyLabelsPath = self.getPolyLabelsPath()
        self.rectLabelsPath = self.getRectLabelsPath()
        self.roi_size = roi_size
        self.t = t

    def getPolyLabelsPath(self):
        return [imagePath.replace('images', 'txtData/polyLabel').replace('.png', '.txt').replace('.jpg', '.txt').replace('.bmp', '.txt') for
                imagePath in self.imagesPath]

    def getRectLabelsPath(self):
        return [imagePath.replace('images', 'txtData/rectLabel').replace('.png', '.txt').replace('.jpg', '.txt').replace('.bmp', '.txt') for
                imagePath in self.imagesPath]

    # 获取图像路径集合
    def getImagePath(self, imageTxtPath):
        with open(imageTxtPath, 'r') as f:
            filesPath = f.readlines()
            return [file.replace('\n', '') for file in filesPath]

    def getboxwh(self, xyxy):
            # 归一化
            box_w, box_h = xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]
            return torch.stack([box_w, box_h], dim=0).T

    def __len__(self):
        return len(self.imagesPath)

    def __getitem__(self, item):
        imagePath = self.imagesPath[item]
        # print(imagePath)
        polyLabelsPath = self.polyLabelsPath[item]
        rectLabelsPath = self.rectLabelsPath[item]
        polyLabels, cls_Label = getPolyLabels(polyLabelsPath)
        polyLabels = torch.FloatTensor(polyLabels[:, 2:])
        image = cv2.imread(imagePath)
        image = torch.FloatTensor(image).permute(2, 0, 1)
        # image = tv.io.read_image(imagePath).type(torch.FloatTensor)[:3, :, :]
        # 通道转换
        c, h, w = image.shape
        if c > 3:
            image = image[:3, :, :]
        elif c == 1:
            image = image.expand(3, -1, -1)
        xyxyBoxes = getBoxInfo(rectLabelsPath, w=w, h=h, t=self.t)
        boxwhTs = self.getboxwh(xyxyBoxes)
        # 归一化处理
        polyLabels = polyLabels_preDeal(polyLabels, whbox=boxwhTs)  # 获取相对
        boxwhTs = self.getboxwh(xyxyBoxes)
        return image, polyLabels, xyxyBoxes, cls_Label, imagePath, boxwhTs
