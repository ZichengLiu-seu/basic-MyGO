import argparse
import logging
import sys

import torch.optim as optim
import torch.cuda
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

from sklearn.model_selection import KFold

from models import MTLModel
from data import Locomotion_Dataset
from scripts.train import train_MTL
from scripts.test import test_MTL
from utils import EarlyStop


def train_and_validate(train_loader, val_loader, model, optimizer, scheduler, early_stop, args):
    train_MTL(args, model=model, train_loader=train_loader, val_loader=val_loader,
              optim=optimizer, scheduler=scheduler, earlystop=early_stop)


def k_fold_cross_validation(k, train_dataset, test_dataset, args):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    ACC = []
    F1 = []
    MSE = []
    MIS = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        logging.info(f'Fold {fold + 1}/{k}')
        train_data = Subset(train_dataset, train_idx)
        val_data = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = MTLModel(args)

        optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.5, total_iters=1000)
        early_stop = EarlyStop(patience=args.patience)

        train_MTL(args, model=model, train_loader=train_loader, val_loader=val_loader,
                  optim=optimizer, scheduler=scheduler, earlystop=early_stop)

        del train_loader, train_data
        torch.cuda.empty_cache()

        acc, f1, mse, mis = test_MTL(args, model, test_loader)
        logging.info(
            "--------------Fold {} Evaluating: acc: {:.4f}% f1: {:.4f} mse: {:.4f} mis: {:.4f}--------------"
            .format(fold + 1, acc, f1, mse, mis))

        ACC.append(acc)
        F1.append(f1)
        MSE.append(mse)
        MIS.append(mis)

    avg_ACC = sum(ACC) / k
    avg_F1 = sum(F1) / k
    avg_MSE = sum(MSE) / k
    avg_MIS = sum(MIS) / k
    logging.info("Feature_MTL Training Results; Learning rate: {}\n"
                 "The Average Acc: {:.4f} %\n"
                 "The Average F1-score: {:.4f}\n"
                 "The Average MSE: {:.4f} m^2\n"
                 "The Average MISDist: {:.4f} m"
                 .format(args.learning_rate, avg_ACC * 100, avg_F1, avg_MSE, avg_MIS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True, default='./data')
    parser.add_argument('--batch_size', type=int, required=True, default=32)
    parser.add_argument('--epochs', type=int, required=True, default=20)
    parser.add_argument('--patience', type=int, required=True, default=6)
    parser.add_argument('--learning_rate', type=float, required=True, default=5e-5)
    parser.add_argument('--checkpoints_path', type=str, required=False, default='checkpoints')
    parser.add_argument('--sup_weight', type=float, required=False, default=0.5)
    parser.add_argument('--interaction_type', type=str, required=True, default='Touchpad')
    parser.add_argument('--model_type', type=str, required=True, default='MTL')

    # args = argparse.Namespace(
    #     root_path='./data', interaction_type='Touchpad', checkpoints_path='checkpoints',
    #     batch_size=32, epochs=30, learning_rate=5e-5, patience=6, model_type="MTL"
    # )
    args = parser.parse_args()

    logpath = 'results/{0}/{1}/output.log'.format(args.model_type, args.interaction_type)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logpath),
            logging.StreamHandler(sys.stdout)
        ]
    )
    full_dataset = Locomotion_Dataset(args.root_path, flag='all', interaction_type=args.interaction_type, scale=True)
    test_dataset = Locomotion_Dataset(args.root_path, flag='test', interaction_type=args.interaction_type, scale=True)
    k_fold_cross_validation(5, full_dataset, test_dataset, args)


if __name__ == '__main__':
    main()
