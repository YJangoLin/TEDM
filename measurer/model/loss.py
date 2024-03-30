# -*- coding: utf-8 -*-
# @Time    : 2024/1/19 14:17
# @Author  : ZL.Liang
# @FileName: loss.py
# @Software: PyCharm
# @Blog    ：https://github.com/YJangoLin
import torch
import torch.nn as nn


class measureLoss(nn.Module):
    def __init__(self, loss_weight):
        super().__init__()
        self.loss_weight = loss_weight

    def circleLoss(self, pred, target):
        n = pred.shape[0]
        center_pred, center_targ = pred[:, :2], target[:, :2]
        R_pred, R_tag = (pred[:, 2] + pred[:, 3]) / 2, target[:, 2]
        centerLoss = torch.sum(abs(center_pred - center_targ)) / n
        RLoss = torch.sum(abs(R_pred - R_tag))*2 / n
        angle_pred, angle_tag = pred[:, 4], target[:, 4]
        angleLoss = torch.sum(abs(angle_pred - angle_tag)) / n
        return self.loss_weight[1]*RLoss + self.loss_weight[0]*centerLoss + self.loss_weight[2]*angleLoss


    def eplliseLoss(self, pred, target):
        n = pred.shape[0]
        center_pred, center_targ = pred[:, :2], target[:, :2]
        R_pred, R_tag = pred[:, 2:4 ], target[:, 2:4]
        angle_pred, angle_tag = pred[:, 4], target[:, 4]
        centerLoss = torch.sum(abs(center_pred - center_targ)) / n
        RLoss = torch.sum(abs(R_pred - R_tag)) / n
        angleLoss = torch.sum(abs(angle_pred - angle_tag)) / n
        return self.loss_weight[1]*RLoss + self.loss_weight[0]*centerLoss + self.loss_weight[2]*angleLoss

    # 设置一个类别损失
    def forward(self, pred, target, cls):
        # 要不要设置类别损失？
        # pred_cls = torch.zeros_like(target[:, 0])
        # 获取长短轴差值, 按照差值将目标分类。差值>0.5认为是椭圆，其他认为是圆(角度按0去计算)
        e_cond = (abs(pred[:, 2] - pred[:, 3]) > 0.5).nonzero()
        # pred_cls[e_cond[:, 0]] = 1
        if len(e_cond) != 0:
            eplliseLoss = self.eplliseLoss(pred[e_cond[:, 0], :], target[e_cond[:, 0], :])
        else:
            eplliseLoss = 0.
        c_cond = (abs(pred[:, 2] - pred[:, 3]) <= 0.5).nonzero()
        pred[c_cond[:, 0], 4] = 0.
        if len(c_cond) != 0:
            circleLoss = self.circleLoss(pred[c_cond[:, 0], :], target[c_cond[:, 0], :])
        else:
            circleLoss = 0.
        # 计算圆的损失：
        return eplliseLoss + circleLoss


# if __name__ == '__main__':
#     a = torch.randint(0, 2, (10, 1))
#     print(a)
#     print(len(a[a == 1]))
