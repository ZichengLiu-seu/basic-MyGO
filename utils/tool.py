import os
import logging
import math

import torch


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logging.info('Updating learning rate to {}'.format(lr))


class EarlyStop:
    def __init__(self, delta=1e-2, patience=3):
        self.min_loss = None
        self.delta = delta
        self.patience = patience
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        if self.min_loss is None:
            self.min_loss = math.inf
            self.save_checkpoint(val_loss, model, path)
            self.min_loss = val_loss
        elif val_loss > self.min_loss - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model, path)
            self.min_loss = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        print(f'Validation loss decreased ({self.min_loss:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))