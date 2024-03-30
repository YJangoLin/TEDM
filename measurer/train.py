# -*- coding: utf-8 -*-
# @Time    : 2024/1/14 10:33
# @Author  : ZL.Liang
# @FileName: train.py
# @Software: PyCharm
# @Blog    ：https://github.com/YJangoLin
import os.path
import yaml
import numpy as np
import torch
import argparse
import torch.nn as nn
from model.model import Measurer_large
from dataset.dataset import MEADataset
from torch.utils.data import DataLoader
from utils.util import collate_fn, polyLabels_back
from model.loss import measureLoss


def save_checkpoint(fileName, model, epoch, optimizer, saveDir):
    state = {}
    state['model'] = model.state_dict()
    state['epoch'] = epoch
    state['optimizer'] = optimizer
    filePath = saveDir + '/' + fileName
    torch.save(state, filePath)


def train(model, data_loader, test_loader, optimizer, loss_fn, epochs, opt, device, save_best=True):
    result_loss = {'epoch': [], 'train_loss':[], 'train_SmoothL1Loss': [],'train_mseloss': [],'train_maeloss': [], 
                    "test_loss": [], 'test_SmoothL1Loss': [],'test_mseloss': [],'test_maeloss': []}
    # state = {}
    for epoch in range(opt['start_epoch'], epochs):
        print('current epoch = {}'.format(epoch))
        train_loss_sum, train_sl1loss_sum, mseloss_sum, maeloss_sum = 0,0, 0, 0
        batch_count = 0
        for i, infoDict in enumerate(data_loader):
            batch_count += 1
            model.train()
            images = infoDict.get('images')
            polyLabels = infoDict.get("polyLabels")
            rectLabels = infoDict.get("rectLabels")  # list[tensor(30, 4), (37, 4)]
            clsLabels = infoDict.get("clsLabels")
            wh = infoDict.get("boxwhs")
            whTs = torch.cat(wh, dim=0).to(device) # (n, 2)
            # 对label进行处理
            polyLabelsTs = torch.cat(polyLabels, dim=0).to(device)  # (67, 5)

            # 使用还原的特征来计算的求法
            rectLabelsTs = torch.cat(rectLabels, dim=0).to(device)  # (67, 4)
            outputs = model(images, rectLabels)  # (67, 5)
            # outputs = polyLabels_back(outputs, whTs, rectLabelsTs)
            cls = torch.cat(clsLabels, dim=0)
            if opt['loss'] == 'ECLoss':
                loss = loss_fn(outputs, polyLabelsTs, cls)  # 计算模型的损失
            elif opt['loss'] == 'SmoothL1Loss':
                axisloss = loss_fn(outputs[:, :2], polyLabelsTs[:, :2])
                xyloss = loss_fn(outputs[:, 2:4], polyLabelsTs[:, 2:4])
                loss = axisloss * opt["loss_weight"][0] + xyloss * opt["loss_weight"][1]
            else:
                loss = loss_fn(outputs, polyLabelsTs)
            
            # polyLabelsTs = polyLabels_back(polyLabelsTs, whTs)
            optimizer.zero_grad()  # 在做反向传播前先清除网络状态
            loss.backward()  # 损失值进行反向传播
            optimizer.step()  # 参数迭代更新
            sl1loss = nn.SmoothL1Loss()(outputs, polyLabelsTs)
            mseloss = nn.MSELoss()(outputs, polyLabelsTs)
            maeLoss = nn.L1Loss()(outputs, polyLabelsTs)
            train_loss_sum += loss.item()
            train_sl1loss_sum += sl1loss.item()  # item()返回的是tensor中的值，且只能返回单个值（标量），不能返回向量，使用返回loss等
            mseloss_sum += mseloss.item()
            maeloss_sum += maeLoss.item()
        test_loss_sum, test_SmoothL1Loss, test_mse_loss, test_mae_loss = evaluate_test(test_loader, model, loss_fn, loss=opt['loss'], loss_weight=opt["loss_weight"])
        print('epoch:{0}/{1},   train_loss: {2:6f}, train_SmoothL1Loss:{3:.6f}, train_mseloss:{4:.6f}, train_maeloss:{5:.6f},  test_loss:{6:.6f},test_SmoothL1Loss:{7:.6f},  test_mseloss:{8:.6f}, test_maeloss:{9:.6f}'.format(
            epoch, epochs, (train_loss_sum / batch_count),train_sl1loss_sum/batch_count,mseloss_sum/batch_count, maeloss_sum/batch_count, test_loss_sum, test_SmoothL1Loss,test_mse_loss, test_mae_loss))
        result_loss.get('epoch').append(epoch)
        result_loss.get('train_loss').append(train_loss_sum / batch_count)
        result_loss.get('train_SmoothL1Loss').append(train_sl1loss_sum / batch_count)
        result_loss.get('train_mseloss').append(mseloss_sum/batch_count)
        result_loss.get('train_maeloss').append( maeloss_sum/batch_count)
        result_loss.get('test_loss').append(test_loss_sum)
        result_loss.get('test_SmoothL1Loss').append(test_SmoothL1Loss)
        result_loss.get('test_mseloss').append(test_mse_loss)
        result_loss.get('test_maeloss').append(test_mae_loss)
        # save
        if not opt['noSave']:
            with open(opt['saveDir'] + "/result.txt", 'a') as f:
                sep = '\t'
                if epoch == 0:
                    resultTxt = ''
                    for indent in result_loss.keys():
                        resultTxt += indent + sep
                    f.write(resultTxt + '\n')
                resultTxt = ''
                for indent in result_loss.keys():
                    resultTxt += str(result_loss.get(indent)[epoch]) + sep
                f.write(resultTxt + '\n')
        if save_best and epoch!=0 and result_loss.get('test_loss')[epoch] < result_loss.get('test_loss')[epoch - 1]:
            save_checkpoint(fileName='best.pth', model=model, epoch=epoch, optimizer=optimizer, saveDir=opt['saveDir'])
        if not opt['noSave'] and (epoch + 1) % opt['saveStep'] == 0:
            save_checkpoint(fileName=f'ckpt_{str(epoch)}.pth', model=model, epoch=epoch, optimizer=optimizer,
                            saveDir=opt['saveDir'])
        if not opt['noSave'] and (epoch + 1) == epochs:
            save_checkpoint(fileName=f'last.pth', model=model, epoch=epoch, optimizer=optimizer,
                            saveDir=opt['saveDir'])
    print('------------finish training-------------')


# 交叉熵损失函数
# loss_fn = torch.nn.CrossEntropyLoss()


def evaluate_test(data_iter, model, loss_fn, device='cuda', loss='SmoothL1Loss', loss_weight=[3, 5]):
    '''
        模型预测精度
    '''
    total = 0
    test_loss_sum, test_sl1loss_sum, test_mseloss_sum, test_maeloss_sum = 0, 0, 0, 0
    with torch.no_grad():
        model.eval()
        for i, infoDict in enumerate(data_iter):
            total += 1
            images = infoDict.get('images')
            polyLabels = infoDict.get("polyLabels")
            rectLabels = infoDict.get("rectLabels")
            clsLabels = infoDict.get("clsLabels")
            wh = infoDict.get("boxwhs")
            whTs = torch.cat(wh, dim=0).to(device)
            clsLabels = torch.cat(clsLabels, dim=0).to(device)
            polyLabelsTs = torch.cat(polyLabels, dim=0).to(device)
            outputs = model(images, rectLabels)  # 5
            # rectLabelsTs = torch.cat(rectLabels, dim=0).to(device)
            # outputs = polyLabels_back(outputs, whTs , rectLabelsTs)
            # outputs = polyLabels_back(rectLabelsTs, outputs)
            if loss == 'ECLoss':
                loss = loss_fn(outputs, polyLabelsTs, clsLabels)  # 计算模型的损失
                # todo: 损失函数改一下考虑MSE，MAE等
            elif loss == 'SmoothL1Loss':
                axisloss = loss_fn(outputs[:, :2], polyLabelsTs[:, :2])
                xyloss = loss_fn(outputs[:, 2:], polyLabelsTs[:, 2:])
                loss = axisloss * loss_weight[0] + xyloss * loss_weight[1]
            else:
                loss = loss_fn(outputs, polyLabelsTs)
            sl1Loss = nn.SmoothL1Loss()(outputs, polyLabelsTs)
            mseloss = nn.MSELoss()(outputs, polyLabelsTs)
            maeLoss = nn.L1Loss()(outputs, polyLabelsTs)
            test_loss_sum += loss.item()
            test_sl1loss_sum += sl1Loss.item()
            test_mseloss_sum += mseloss.item()
            test_maeloss_sum += maeLoss.item()
    return test_loss_sum / total, test_sl1loss_sum/total, test_mseloss_sum/total, test_maeloss_sum/total



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='SD2',
                        help='project name')
    parser.add_argument('--TrainPath', type=str, default='F:/TEDM/dataset/Satellite Dataset#2/train.txt',
                        help='dataset path')
    parser.add_argument('--valPath', type=str, default='F:/TEDM/dataset/Satellite Dataset#2/val.txt',
                        help='dataset path')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--manual_seed', type=int, default=1, help='manual seed')
    parser.add_argument('--in_dim', type=int, default=3, help='image input dim')
    parser.add_argument('--output_dim', type=int, default=3, help='image input dim')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='')
    # 
    parser.add_argument('--t', type=int, default=0, help='')
    parser.add_argument('--roi_size', type=int, default=128, help='')

    parser.add_argument('--noSave', action='store_true', help='')
    parser.add_argument('--saveStep', type=int, default=10, help='')
    parser.add_argument('--saveDir', type=str, default='./checkpoints', help='')
    parser.add_argument('--loss', type=str, default='SmoothL1Loss', choices=['ECLoss', 'MSELoss', 'SmoothL1Loss'])
    parser.add_argument('--loss_weight', type=list, default=[3, 5],
                        help='the weight of centerLoss, aixsLoss, angleLoss')
    parser.add_argument('--inConv', action='store_true',
                        help='do you use inConv?')
    opt = parser.parse_args()
    opt = vars(opt)
    torch.manual_seed(opt['manual_seed'])
    if not opt['noSave']: opt['saveDir'] = opt['saveDir'] + '/' + opt['name']
    if not os.path.exists(opt['saveDir']):
        os.makedirs(opt['saveDir'])
    # 训练轮数
    model = Measurer_large(opt=opt).to(opt['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
    if opt['resume']:
        checkpoint = torch.load(opt['resume'], map_location=torch.device(opt['device']))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        opt['start_epoch'] = checkpoint['epoch'] + 1
    # 优化算法Adam = RMSProp + Momentum (梯度、lr两方面优化下降更快更稳)
    if opt['loss'] == 'MSELoss':
        loss_fn = nn.MSELoss()
    elif opt['loss'] == 'ECLoss':
        loss_fn = measureLoss(opt['loss_weight'])
    elif opt['loss'] == 'SmoothL1Loss':
        loss_fn = nn.SmoothL1Loss(opt['loss_weight'])
    # 创建dataset
    trainDataset = MEADataset(opt['TrainPath'], roi_size=opt['roi_size'], t=opt["t"])
    valDataset = MEADataset(opt['valPath'], roi_size=opt['roi_size'], t=opt["t"])
    train_dataloader = DataLoader(trainDataset, batch_size=opt["batch_size"], shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(valDataset, batch_size=opt["batch_size"], shuffle=True, collate_fn=collate_fn)
    # 将字典转换为YAML格式并写入文件
    with open(f'{opt["saveDir"]}/opt.yml', 'w') as file:
        yaml.dump(opt, file)
    # 创建dataset，dataloader
    train(model, optimizer=optimizer, loss_fn=loss_fn, data_loader=train_dataloader,
          test_loader=test_dataloader,
          epochs=opt['epochs'], device=opt['device'], opt=opt)



if __name__ == '__main__':
    main()
