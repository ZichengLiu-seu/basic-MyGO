import os
import logging

import torch

from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.lossupdate import EpochLoss
from models.criterion import MTLLoss


def train_MTL(args, model, train_loader, val_loader, optim, scheduler, earlystop):
    train_loss = EpochLoss()
    train_cls_loss = EpochLoss()
    train_reg_loss = EpochLoss()
    val_loss = EpochLoss()
    val_cls_loss = EpochLoss()
    val_reg_loss = EpochLoss()
    criterion = MTLLoss()

    for epoch in range(args.epochs):
        if args.process_display:
            train_bar = tqdm(train_loader)
            val_bar = tqdm(val_loader)
        else:
            train_bar = train_loader
            val_bar = val_loader
        model.zero_grad()
        model.train()
        for train_input in train_bar:
            batch_x, batch_y, batch_dirt = map(lambda x: x.cuda(), train_input)
            pred_cls, pred_reg = model(batch_x)
            clsloss, regloss, loss = criterion(pred_cls, pred_reg, batch_dirt, batch_y)

            train_loss.update(loss.item())
            train_cls_loss.update(clsloss.item())
            train_reg_loss.update(regloss.item())

            criterion.adjust_alpha(clsloss.item(), regloss.item())

            loss.backward()
            optim.step()
            model.zero_grad()
        train_loss.record()
        train_cls_loss.record()
        train_reg_loss.record()

        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            for val_input in val_bar:
                batch_x, batch_y, batch_dirt = map(lambda x: x.cuda(), val_input)
                pred_cls, pred_reg = model(batch_x)
                clsloss, regloss, loss = criterion(pred_cls, pred_reg, batch_dirt, batch_y)

                val_loss.update(loss.item())
                val_cls_loss.update(clsloss.item())
                val_reg_loss.update(regloss.item())
            val_loss.record()
            val_cls_loss.record()
            val_reg_loss.record()

        torch.cuda.empty_cache()
        scheduler.step()
        if epoch % 10 == 0:
            logging.info('Epoch:{} ; Train Loss:{:.4f} ; Val Loss:{:.4f}\n'
                         '     Details:\n'
                         '     Train: Classify Loss:{:.4f} ; Regression Loss:{:.4f}\n'
                         '     Valid: Classify Loss:{:.4f} ; Regression Loss:{:.4f}\n'
                         '     Classification Alpha:{:.6f}\n'
                         .format(epoch, train_loss.loss[-1], val_loss.loss[-1],
                                 train_cls_loss.loss[-1], train_reg_loss.loss[-1],
                                 val_cls_loss.loss[-1], val_reg_loss.loss[-1],
                                 criterion.alpha))

        earlystop(val_loss.avg, model, args.checkpoints_path)
        if earlystop.early_stop:
            logging.info("EarlyStopping!")
            break

    loss_graph(args, train_cls_loss.loss, val_cls_loss.loss, epoch, "cls")
    loss_graph(args, train_reg_loss.loss, val_reg_loss.loss, epoch, "reg")
    loss_graph(args, train_loss.loss, val_loss.loss, epoch, "total")


def loss_graph(args, train_loss, val_loss, epoch, name):
    plt.figure()
    plt.plot(range(epoch + 1), train_loss, label='train loss')
    plt.plot(range(epoch + 1), val_loss, label='val loss')
    plt.legend()

    path = 'results/{0}/{1}'.format(args.model_type, args.interaction_type)
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path + '/{} loss.png'.format(name))
