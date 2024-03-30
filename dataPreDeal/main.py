# -*- coding: utf-8 -*-
# @Time    : 2024/1/20 19:31
# @Author  : ZL.Liang
# @FileName: main.py.py
# @Software: PyCharm
# @Blog    ：https://github.com/YJangoLin
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from DataSwitch import buildXml
from VocToYolo import buildTxt
from dataSplit import splitData


def main():
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        opt = yaml.safe_load(f)
    buildXml(imageDir=opt['imagesDir'], jsonLabelDir=opt['labelsDir'], outputXml=opt['dataPreDeal']['outputXml'])
    buildTxt(inputXmlDir=opt['dataPreDeal']['outputXml'], outputTxtDir=opt['dataPreDeal']['outputTxt'])
    splitData(imageDir=opt['imagesDir'], outputPath=opt['dataPreDeal']['splitTxtPath'],
              trainRadio=opt['dataPreDeal']['trainRadio'])


if __name__ == '__main__':
    main()
