# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 15:23
# @Author  : ZL.Liang
# @FileName: main.py.py
# @Software: PyCharm
# @Blog    ：https://github.com/YJangoLin
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'yolov7'))
sys.path.append(os.path.join(os.getcwd(), 'measurer'))

import yaml

if __name__ == '__main__':
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        opt = yaml.safe_load(f)
    sep = " "
    print(opt)
    detectCMD = f'python yolov7/detect.py --device {opt["detector"]["device"]}  --project {opt["detector"]["output"]} --weights "{opt["detector"]["weight"]}" --source "{opt["detector"]["source"]}" --name {opt["detector"]["name"]}'
    if opt['detector']['saveTxt']:
        detectCMD += sep + "--save-txt"
    if opt['detector']['val']:
        detectCMD += sep + f'--val --valTxtPath "{opt["detector"]["valTxtPath"]}"'
    print(detectCMD)
    os.system(detectCMD)
    measureCMD = f'python measurer/measure.py --name {opt["measurer"]["name"]} --device {opt["measurer"]["device"]}  --rectDir "{opt["measurer"]["rectDir"]}" --imageDir "{opt["measurer"]["imageDir"]}" --weights "{opt["measurer"]["weight"]}" --saveDir {opt["measurer"]["output"]}'
    if opt['measurer']['val']:
        measureCMD += sep + f'--val --valTxtPath "{opt["detector"]["valTxtPath"]}"'
    if opt['measurer']['saveTxt']:
        measureCMD += sep + "--save-txt"
    if opt['measurer']['config']:
        measureCMD += sep + f'--config "{opt["measurer"]["config"]}"'
    print(measureCMD)
    os.system(measureCMD)


