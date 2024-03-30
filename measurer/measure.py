import os
import time
import numpy as np
import shutil
from model.mobileNet import mobilenet_v3_large
import torch
import torchvision as tv
from PIL import Image, ImageDraw
import cv2
import argparse
import yaml
from utils.util import getboxwh, polyLabels_back, xywhn2xyxy


def read_imageandrectLabel(imagePath, rectPath, device='cuda'):
    with open(rectPath, 'r') as f:
        rectLabels = np.array([row.replace('\n', '').split(" ") for row in f.readlines()], dtype=float)
    image = cv2.imread(imagePath)
    image = torch.FloatTensor(image).permute(2, 0, 1)
    return image.unsqueeze(0).to(device), torch.FloatTensor(rectLabels).to(device)


def read_polyLabel(polyPath, device='cuda'):
    with open(polyPath, 'r') as f:
        polyLabels = np.array([row.replace('\n', '').split(" ")[1:] for row in f.readlines()], dtype=float)  # 把种类去除
    return torch.FloatTensor(polyLabels).to(device)


def draw_arc(imagePath, polyLabels, outputFilePath, axisLong_xy):
    polyLabels = polyLabels.detach().cpu().numpy()
    image = cv2.imread(imagePath)
    for ind, polyLabel in enumerate(polyLabels):
        # a, b为半短轴长
        image = cv2.ellipse(image, (int(polyLabel[0]), int(polyLabel[1])), (int(polyLabel[2]), int(polyLabel[3])),
                            round(polyLabel[4], 0), 0, 360,
                            (0, 255, 0),
                            2)
        # print((int(axisLong_xy[ind][0]), int(axisLong_xy[ind][1])))
        image = cv2.putText(image, str(ind+1), (int(axisLong_xy[ind][0]), int(axisLong_xy[ind][1])),fontFace=cv2.FONT_HERSHEY_SIMPLEX ,color=(0, 0, 255) ,fontScale=1)
        image = cv2.circle(image, (int(axisLong_xy[ind][0]), int(axisLong_xy[ind][1])),2, (255, 0, 0) ,thickness=-1)
    # cv2.imshow("image", image)
    cv2.imwrite(outputFilePath, image)
    # cv2.ellipse(img, (60, 20), (60, 20), 0, 0, 360, (255, 255, 255), 2);
    cv2.waitKey()
    cv2.destroyAllWindows()


def eval(model, src, boxes):
    model.eval()
    with torch.no_grad():
        output = model(src, boxes)
    return output

def get_angle(axisLongXY):
    angleRad = torch.atan2(axisLongXY[:, 1] , axisLongXY[:, 0])
    return torch.rad2deg(angleRad)

def get_epllise_info(xyxybox, whbox, pred):
    angle = get_angle(pred[:, 2:])
    polyLabels = polyLabels_back(pred, whbox, xyxybox)
    # angle = get_angle(polyLabels[:, 2:])
    center_x, center_y = (xyxybox[:, 2] + xyxybox[:, 0])/2, (xyxybox[:, 3] + xyxybox[:, 1])/2
    # print(f'axisXY:{polyLabels[:, 2:]}')
    return torch.stack([center_x, center_y, polyLabels[:, 0], polyLabels[:, 1], angle], dim=0).T, polyLabels[:, 2:]


def measure(imagePath, rectTxt, model, opt):
    imageTs, rectLabelTs = read_imageandrectLabel(imagePath, rectTxt, opt['device'])
    xywhboxes = rectLabelTs[:, 1:]
    _, _, h, w = imageTs.shape
    xyxyboxes = xywhn2xyxy(xywhboxes, h=h, w=w)
    whBox = getboxwh(xyxyboxes)
    xyxyboxesList = [xywhn2xyxy(xywhboxes)]
    pred = eval(model, imageTs, xyxyboxesList)
    epllise_pred_gt, axisLong_xy = get_epllise_info(xyxyboxes, whBox, pred)
    fileName = imagePath.split('/')[-1]
    if opt['save']:
        draw_arc(imagePath, epllise_pred_gt, outputFilePath=opt['saveDir'] + '/' + fileName, axisLong_xy=axisLong_xy)
    if opt['save_txt']:
        txtName = fileName.split('.')[0] + '.txt'
        if not os.path.exists(opt['saveDir'] + '/labels/'): os.makedirs(opt['saveDir'] + '/labels/')
        np_data = epllise_pred_gt.cpu().numpy()
        print(opt['saveDir'])
        np.savetxt(opt['saveDir'] + '/' + 'labels/' + txtName, np_data, fmt='%.4f', delimiter=' ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='ConcentricED', help='')
    parser.add_argument('--imageDir', type=str, default='./data/images', help='')
    parser.add_argument('--rectDir', type=str, default='F:/TEDM/dataset/Concentric Ellipses Dataset/txtData/rectLabel')
    parser.add_argument('--config', type=str, default='./checkpoints/ConcentricED/opt.yml')
    parser.add_argument('--weights', type=str, default='./checkpoints/ConcentricED/best.pth', help='')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--device', type=str, default='cuda', help='')
    parser.add_argument('--save', type=bool, default=True, help='')
    parser.add_argument('--saveDir', type=str, default='./checkpoints', help='')
    parser.add_argument('--loss_weight', type=list, default=[3, 5],
                        help='the weight of centerLoss, aixsLoss, angleLoss')
    parser.add_argument('--val', action='store_true', help='Is source the Dir of valDataset')
    parser.add_argument('--valTxtPath', type=str, default='./val.txt', help='Is source the Dir of valDataset')
    opt = parser.parse_args()
    opt = vars(opt)
    opt["saveDir"] = f"{opt['saveDir']}/{opt['name']}"
    if not os.path.exists(opt["saveDir"]): os.makedirs(opt["saveDir"])
    if opt['val']:
        with open(opt['valTxtPath'], 'r') as f:
            imagesPaths = [row.replace('\n', '') for row in f.readlines()]
            rectLabelPaths = []
            for imagePath in imagesPaths:
                imageType = imagePath.split('/')[-1].split('.')[-1]
                # rectName = fileName.split('.')[0] + '.txt'
                rectSource = imagePath.replace('images', 'txtData/rectLabel').replace(f'.{imageType}', '.txt')
                rectLabelPaths.append(rectSource)
                # shutil.copy(imagePath, opt['imageDir'] + '/' + fileName)
                # shutil.copy(rectSource, opt['rectDir'] + '/' + rectName)
    else: 
        # 
        imagesPaths = [opt['imageDir']+'/' + fileName for fileName in os.listdir(opt['imageDir'])]
        rectLabelPaths = []
        for imagePath in imagesPaths:
            fileName = imagePath.split('/')[-1]
            rectName = fileName.split('.')[0] + '.txt'
            rectLabelPaths.append(opt['rectDir'] + '/' + rectName)
    # load model
    state = torch.load(opt['weights'], map_location=torch.device(opt['device']))
    print(state.keys())
    with open(opt['config'], 'r') as f:
        config =  yaml.safe_load(f)
    config['device'] = opt['device']
    model = mobilenet_v3_large(opt=config).to(opt['device'])
    model.load_state_dict(state['model'])
    if not os.path.exists(opt['saveDir']): os.makedirs(opt['saveDir'])
    for ind, imagesPath in enumerate(imagesPaths):
        start = time.time()
        measure(imagesPath, rectLabelPaths[ind], model, opt=opt)
        end = time.time()
        print(f"{imagesPath} run time {str((end - start)*1000)} ms")
