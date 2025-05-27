import os
import logging

import torch

from tqdm import tqdm

from utils.result_vis import pred_spilt, route_1d, route_2d
from models.criterion import MTLEvaluate


def test_MTL(args, model, test_loader):
    criterion = MTLEvaluate()
    model.cuda()

    if args.process_display:
        test_bar = tqdm(test_loader)
    else:
        test_bar = test_loader

    for test_input in test_bar:
        batch_x, batch_y, batch_dirt = map(lambda x: x.cuda(), test_input)
        model.eval()
        pred_cls, pred_reg = model(batch_x)

        criterion(pred_cls, pred_reg, batch_dirt, batch_y)

    acc, f1, mse, mis = criterion.Acc()
    logging.info("Learning rate: {}\n"
                 "The accuracy: {:.4f} %\n"
                 "The Average F1-score: {:.4f}\n"
                 "The Average MSE: {:.4f} m^2\n"
                 "The Average MISDist: {:.4f} m"
                 .format(args.learning_rate, acc * 100, f1, mse, mis))
    pred_x, pred_y, true_x, true_y = pred_spilt(criterion.pred, criterion.gt,
                                                True)
    route_1d(args, pred_x, pred_y, true_x, true_y)
    route_2d(args, pred_x, pred_y, true_x, true_y)

    return acc, f1, mse, mis
