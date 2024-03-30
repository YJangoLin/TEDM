import numpy as np

from model.model import Measurer_large
import torch
import torchvision as tv
from utils.util import polyLabels_back, xywhn2xyxy
from PIL import Image,ImageDraw
import cv2

def read_imageandrectLabel(imagePath, rectPath, device='cuda'):
    with open(rectPath, 'r') as f:
        rectLabels = np.array([row.replace('\n', '').split(" ") for row in f.readlines()], dtype=float)
    return tv.io.read_image(imagePath).type(torch.FloatTensor).unsqueeze(0).to(device), torch.FloatTensor(rectLabels).to(device)

def read_polyLabel(polyPath, device='cuda'):
    with open(polyPath, 'r') as f:
        polyLabels = np.array([row.replace('\n', '').split(" ")[1:] for row in f.readlines()], dtype=float) # 把种类去除
    return torch.FloatTensor(polyLabels).to(device)


def draw_arc(imagePath, polyLabels, outputFilePath):
    polyLabels = polyLabels.detach().cpu().numpy()
    image = cv2.imread(imagePath)
    for polyLabel in polyLabels:
    # a, b为半短轴长
        image = cv2.ellipse(image, (int(polyLabel[0]), int(polyLabel[1])), (int(polyLabel[2]), int(polyLabel[3])), round(polyLabel[4], 0), 0, 360,
                            (0, 255, 0),
                            2)
    cv2.imshow("image", image)
    cv2.imwrite(outputFilePath, image)
    # cv2.ellipse(img, (60, 20), (60, 20), 0, 0, 360, (255, 255, 255), 2);
    cv2.waitKey()
    cv2.destroyAllWindows()

def eval(model, src, boxes, cls):
    model.eval()
    with torch.no_grad():
        output = model(src, boxes, cls)
    return polyLabels_back(xyxy=boxes[0], polyLabels=output)

rectPath = './data/rect_train8.txt'
imagePath = './data/train8.png'
device = 'cuda'
imageTs, rectLabelTs = read_imageandrectLabel(imagePath, rectPath)
clsTs = [rectLabelTs[:, 0]]
xuwhboxes = rectLabelTs[:, 1:]
xyxyboxes = [xywhn2xyxy(xuwhboxes)]
checkpointPath = './checkpoints/checkpoints_300.pth'
state = torch.load(checkpointPath)
print(state.keys())
opt = state['config']
print(opt)
model = Measurer_large(opt=opt).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
model.load_state_dict(state['model_state'])
optimizer.load_state_dict(state['optimizer'])
pred = eval(model, imageTs, xyxyboxes, clsTs)


polyFilePath = './data/poly_train8.txt'
draw_arc(imagePath, read_polyLabel(polyFilePath), outputFilePath='train8_targ.png')
draw_arc(imagePath, pred, outputFilePath='train8_pred.png')


