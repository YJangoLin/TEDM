# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 20:09
# @Author  : ZL.Liang
# @FileName: dataSplit.py
# @Software: PyCharm
# @Blog    ：https://github.com/YJangoLin
import os
import shutil


def splitData(imageDir: str, outputPath, trainRadio=0.8):
    imageDir = os.path.join(os.getcwd(), imageDir)
    fileList = os.listdir(imageDir)
    trainNumber = int(len(fileList) * 0.8)
    with open(outputPath + '/' + 'train.txt', 'w', encoding='utf-8') as f:
        for file in fileList[: trainNumber]:
            filePath = os.path.join(imageDir, file).replace('\\', '/') + '\n'
            f.write(filePath)
    with open(outputPath + '/' + 'val.txt', 'w', encoding='utf-8') as f:
        for file in fileList[trainNumber:]:
            filePath = os.path.join(imageDir, file).replace('\\', '/') + '\n'
            f.write(filePath)
    shutil.copy(outputPath + '/' + 'train.txt', 'dataset/train.txt')
    shutil.copy(outputPath + '/' + 'val.txt', 'dataset/val.txt')


if __name__ == '__main__':
    imageDir = os.path.join(os.getcwd(), 'data/images')
    splitData(imageDir, 0.8)
