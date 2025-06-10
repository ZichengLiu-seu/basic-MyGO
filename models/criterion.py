import logging

import torch
from torch.nn.functional import mse_loss, binary_cross_entropy


class MTLLoss:
    def __init__(self):
        self.alpha = 0.5

    def adjust_alpha(self, loss_task1, loss_task2, beta=0.9):
        ratio = loss_task1 / (loss_task2 + 1e-8)
        self.alpha = beta * self.alpha + (1 - beta) * ratio
        self.alpha = max(0, min(1, self.alpha))

    def __call__(self, x, y, label_x, label_y):
        x = x.squeeze(dim=1)
        clsloss = binary_cross_entropy(x, label_x)
        regloss = mse_loss(y, label_y)
        totalloss = self.alpha * clsloss + (1 - self.alpha) * regloss
        return clsloss, regloss, totalloss


class MSELoss:
    def __init__(self):
        return

    def __call__(self, x, label_x):
        regloss = mse_loss(x, label_x)
        return regloss


class MTLEvaluate:
    def __init__(self):
        self.count = 0

        self.accuracy = 0
        self.f1 = 0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

        self.imse = 0
        self.amse = 0
        self.imis = 0
        self.amis = 0

        self.pred = []
        self.gt = []

    def __call__(self, pred_cls, pred_reg, batch_dirt, batch_y):
        self.count += len(pred_cls)

        pred_labels = (pred_cls > 0.5).int().squeeze()
        true_labels = batch_dirt.int()
        self.tp += torch.sum((pred_labels == 1) & (true_labels == 1)).item()
        self.fp += torch.sum((pred_labels == 1) & (true_labels == 0)).item()
        self.tn += torch.sum((pred_labels == 0) & (true_labels == 0)).item()
        self.fn += torch.sum((pred_labels == 0) & (true_labels == 1)).item()

        self.imse += mse_loss(pred_reg, batch_y, reduction="sum")
        self.imis += torch.sum(torch.norm(pred_reg - batch_y))

        self.pred.extend(pred_reg[i] for i in range(pred_reg.size(0)))
        self.gt.extend(batch_y[i] for i in range(batch_y.size(0)))

    def Acc(self):
        self.accuracy = (self.tp + self.tn) / self.count
        precision = self.tp / (self.tp + self.fp + 1e-10)
        recall = self.tp / (self.tp + self.fn + 1e-10)
        self.f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        self.amse = self.imse / self.count
        self.amis = self.imis / self.count
        logging.info('--------------num:{}; tp:{}; tn:{}; precision:{:.4f}; recall:{:.4f}---------------------'.format(
            self.count, self.tp, self.tn, precision, recall))
        return self.accuracy, self.f1, self.amse, self.amis


class SimpleEvaluate:
    def __init__(self):
        self.count = 0

        self.accuracy = 0

        self.imse = 0
        self.amse = 0
        self.imis = 0
        self.amis = 0

        self.pred = []
        self.gt = []

    def __call__(self, pred_reg, batch_y):
        self.count += len(pred_reg)

        self.imse += mse_loss(pred_reg, batch_y, reduction="sum")
        self.imis += torch.sum(torch.norm(pred_reg - batch_y))

        self.pred.extend(pred_reg[i] for i in range(pred_reg.size(0)))
        self.gt.extend(batch_y[i] for i in range(batch_y.size(0)))

    def Acc(self):
        self.amse = self.imse / self.count
        self.amis = self.imis / self.count
        return self.amse, self.amis
