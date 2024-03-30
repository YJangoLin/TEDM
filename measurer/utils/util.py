# -*- coding: utf-8 -*-
# @Time    : 2024/1/16 14:29
# @Author  : ZL.Liang
# @FileName: util.py
# @Software: PyCharm
# @Blog    ：https://github.com/YJangoLin
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torchvision as tv


# def get_CD(xyxy):
#     xyxy[:, 3:] - xyxy[:, ]

def xywhn2xyxy(x, w=1024, h=768, t=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) - t   # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) - t  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + t  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + t # bottom right y
    return y

# w,h
def getBoxInfo(recLabelFile, w=1024, h=768, t=0):
    xywhData = []
    with open(recLabelFile, 'r') as f:
        rectInfo = f.readlines()
        rectInfo = [rect.replace('\n', '') for rect in rectInfo]
        for rect in rectInfo:
            box = [float(re) for re in rect.split(' ')[1:]]
            xywhData.append(box)  # 去除label
    xywhData = torch.Tensor(xywhData)
    xyxyTensor = xywhn2xyxy(xywhData, w, h, t)  # 获取候选框的左上右下角坐标
    return xyxyTensor


def getEpillInfo(recLabelFile):
    EpillInfo = []
    with open(recLabelFile, 'r') as f:
        rectInfo = f.readlines()
        rectInfo = [rect.replace('\n', '') for rect in rectInfo]
        for rect in rectInfo:
            info = [float(re) for re in rect.split(' ')[1:]]
            EpillInfo.append(info)  # 去除label
    return torch.Tensor(EpillInfo)


def getImageData(image, boxs, pooler):  # roi_align
    objectTs = pooler(image, [boxs])
    return objectTs



# 获取中心点相对偏移,max为偏移量的一个最大值，用于归一化处理, a,b最大值
def polyLabels_preDeal(polyLabels, whbox):
    # 取每个矩形框的对角线的一半
    n = polyLabels.shape[0]
    polyPreDeal = torch.zeros((n, 4))
    angle = torch.deg2rad(polyLabels[:, -1])
    w, h = whbox[:, 0], whbox[:, 1]
    tg = torch.sqrt(w **2 + h **2) /2
    polyPreDeal[:, :2] = polyLabels[:, :2] / tg.unsqueeze(1)
    # x轴坐标为【0， 1】
    polyPreDeal[:, 2] =  polyLabels[:, 0] * torch.cos(angle) / (w /2)
    # y轴坐标范围【-1， 1】
    polyPreDeal[:, 3] = polyLabels[:, 0] * torch.sin(angle) / (h /2)
    return polyPreDeal


def polyLabels_back(output, whbox, xyxy):
    center_x, center_y = (xyxy[:, 0] + xyxy[:, 2])/2, (xyxy[:, 1] + xyxy[:, 3])/2
    w, h = whbox[:, 0], whbox[:, 1]
    tg = torch.sqrt(w **2 + h **2) /2
    output[:, :2] = output[:, :2]* tg.unsqueeze(1)
    # x轴坐标为【0， 1】
    output[:, 2] = output[:, 2] * (w/2) + center_x
    # x轴坐标范围【-1， 1】
    output[:, 3] = output[:, 3] * (h/2) + center_y
    return output


def getboxwh(xyxy):
    box_w, box_h = xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]
    return torch.stack([box_w, box_h], dim=0).T


def draw_arc(imagePath, cp, axis, angle):
    '''
    :param imagePath: image Path
    :param cp: center point (x, y)
    :param axis: long axis and short axis (la, sa)
    :param angle: 偏移角度
    :return: none
    '''
    image = cv2.imread(imagePath)
    # a, b为半短轴长
    image = cv2.ellipse(image, (int(cp[0]), int(cp[1])), (int(axis[0]), int(axis[1])), round(angle, 0), 0, 360,
                        (255, 255, 255),
                        1)
    image = cv2.circle(image, (int(cp[0]), int(cp[1])), 2, color=(255, 0, 0))
    cv2.imshow("image", image)
    # cv2.ellipse(img, (60, 20), (60, 20), 0, 0, 360, (255, 255, 255), 2);
    cv2.waitKey()
    cv2.destroyAllWindows()


def show_align_result(objectTs: torch.Tensor):
    '''
    :param objectTs:  shape(1, c, h, w)
    :return:
    '''
    img = objectTs.type(torch.int).numpy()
    img = np.transpose(img.squeeze(), (1, 2, 0))
    plt.imshow(img)
    plt.show()



def getAllImageData(imageDir):
    return [imageDir + '/' + imageName for imageName in os.listdir(imageDir)]


def getPolyLabels(polyLabelFile):
    with open(polyLabelFile, 'r') as f:
        context = f.readlines()
        contextArr = np.array([con.replace('\n', '').split(' ')[1:] for con in context]).astype(dtype=float)
        clsArr = np.array([con.replace('\n', '').split(' ')[0] for con in context]).astype(dtype=float)
        return torch.Tensor(contextArr), torch.Tensor(clsArr)


def collate_fn(batch):  # batch是DataLoader传进来的，相当于是getitem的结果放到一个元组里，这个元组里有batch_size个元素  ([imgs,labels],...)
    # 自己定义怎么整理数据, list[output1, output2]
    # 对image进行stack
    images, polyLabels, xyxyBoxes, cls_Labels, imagePaths, boxwhs = [], [], [], [], [], []
    for b in batch:
        image, polyLabel, xyxyBox, cls_Label, imagePath, wh = b
        images.append(image)
        polyLabels.append(polyLabel)
        xyxyBoxes.append(xyxyBox)
        cls_Labels.append(cls_Label)
        imagePaths.append(imagePath)
        boxwhs.append(wh)
        # polyLabels = torch.stack([polyLabels, polyLabel], dim=0)
    # 对poly
    return {'images': images, 'polyLabels': polyLabels, 'rectLabels': xyxyBoxes, 'clsLabels': cls_Labels, 'imagePaths': imagePaths, 'boxwhs': boxwhs}