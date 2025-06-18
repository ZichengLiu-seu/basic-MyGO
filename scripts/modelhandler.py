import logging

from abc import ABC, abstractmethod

from models import MTLModel, LSTMModel, RegModel
from scripts.train import train_MTL, train_LSTM, train_Reg
from scripts.test import test_MTL, test_LSTM, test_Reg


class ModelHandler(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def train(self, model, train_loader, val_loader, optimizer, scheduler, early_stop):
        pass

    @abstractmethod
    def test(self, model, test_loader):
        pass

    @abstractmethod
    def log_results(self, metrics, fold):
        pass


class MTLHandler(ModelHandler):
    def __init__(self, args):
        super().__init__(args)

    def create_model(self):
        return MTLModel().to(self.args.device)

    def train(self, model, train_loader, val_loader, optimizer, scheduler, early_stop):
        train_MTL(self.args, model=model, train_loader=train_loader, val_loader=val_loader,
                  optim=optimizer, scheduler=scheduler, earlystop=early_stop)

    def test(self, model, test_loader):
        return test_MTL(self.args, model, test_loader)

    def log_results(self, metrics, fold):
        acc, f1, mse, mis = metrics
        logging.info(
            "--------------Fold {} Evaluating: acc: {:.4f}% f1: {:.4f} mse: {:.4f} mis: {:.4f}--------------"
            .format(fold + 1, acc * 100, f1, mse, mis))


class LSTMHandler(ModelHandler):
    def __init__(self, args):
        super().__init__(args)

    def create_model(self):
        return LSTMModel().to(self.args.device)

    def train(self, model, train_loader, val_loader, optimizer, scheduler, early_stop):
        train_LSTM(self.args, model=model, train_loader=train_loader, val_loader=val_loader,
                   optim=optimizer, scheduler=scheduler, earlystop=early_stop)

    def test(self, model, test_loader):
        return test_LSTM(self.args, model, test_loader)

    def log_results(self, metrics, fold):
        mse, mis = metrics
        logging.info(
            "--------------Fold {} Evaluating: mse: {:.4f} mis: {:.4f}--------------"
            .format(fold + 1, mse, mis))


class RegHandler(ModelHandler):
    def __init__(self, args):
        super().__init__(args)

    def create_model(self):
        return RegModel().to(self.args.device)

    def train(self, model, train_loader, val_loader, optimizer, scheduler, early_stop):
        train_Reg(self.args, model=model, train_loader=train_loader, val_loader=val_loader,
                  optim=optimizer, scheduler=scheduler, earlystop=early_stop)

    def test(self, model, test_loader):
        return test_Reg(self.args, model, test_loader)

    def log_results(self, metrics, fold):
        mse, mis = metrics
        logging.info(
            "--------------Fold {} Evaluating: mse: {:.4f} mis: {:.4f}--------------"
            .format(fold + 1, mse, mis))


class LSTMplusHandler(ModelHandler):
    def __init__(self, args):
        super().__init__(args)

    def create_model(self):
        return LSTMplusModel().to(self.args.device)

    def train(self, model, train_loader, val_loader, optimizer, scheduler, early_stop):
        train_MTL(self.args, model=model, train_loader=train_loader, val_loader=val_loader,
                  optim=optimizer, scheduler=scheduler, earlystop=early_stop)

    def test(self, model, test_loader):
        return test_MTL(self.args, model, test_loader)

    def log_results(self, metrics, fold):
        acc, f1, mse, mis = metrics
        logging.info(
            "--------------Fold {} Evaluating: acc: {:.4f}% f1: {:.4f} mse: {:.4f} mis: {:.4f}--------------"
            .format(fold + 1, acc * 100, f1, mse, mis))


class MTLTransHandler(ModelHandler):
    def __init__(self, args):
        super().__init__(args)

    def create_model(self):
        return MTLTransModel().to(self.args.device)

    def train(self, model, train_loader, val_loader, optimizer, scheduler, early_stop):
        train_MTL(self.args, model=model, train_loader=train_loader, val_loader=val_loader,
                  optim=optimizer, scheduler=scheduler, earlystop=early_stop)

    def test(self, model, test_loader):
        return test_MTL(self.args, model, test_loader)

    def log_results(self, metrics, fold):
        acc, f1, mse, mis = metrics
        logging.info(
            "--------------Fold {} Evaluating: acc: {:.4f}% f1: {:.4f} mse: {:.4f} mis: {:.4f}--------------"
            .format(fold + 1, acc * 100, f1, mse, mis))


class MTLLstmHandler(ModelHandler):
    def __init__(self, args):
        super().__init__(args)

    def create_model(self):
        return MTLLstmModel().to(self.args.device)

    def train(self, model, train_loader, val_loader, optimizer, scheduler, early_stop):
        train_MTL(self.args, model=model, train_loader=train_loader, val_loader=val_loader,
                  optim=optimizer, scheduler=scheduler, earlystop=early_stop)

    def test(self, model, test_loader):
        return test_MTL(self.args, model, test_loader)

    def log_results(self, metrics, fold):
        acc, f1, mse, mis = metrics
        logging.info(
            "--------------Fold {} Evaluating: acc: {:.4f}% f1: {:.4f} mse: {:.4f} mis: {:.4f}--------------"
            .format(fold + 1, acc * 100, f1, mse, mis))


class MTLRevHandler(ModelHandler):
    def __init__(self, args):
        super().__init__(args)

    def create_model(self):
        return MTLRevModel().to(self.args.device)

    def train(self, model, train_loader, val_loader, optimizer, scheduler, early_stop):
        train_MTL(self.args, model=model, train_loader=train_loader, val_loader=val_loader,
                  optim=optimizer, scheduler=scheduler, earlystop=early_stop)

    def test(self, model, test_loader):
        return test_MTL(self.args, model, test_loader)

    def log_results(self, metrics, fold):
        acc, f1, mse, mis = metrics
        logging.info(
            "--------------Fold {} Evaluating: acc: {:.4f}% f1: {:.4f} mse: {:.4f} mis: {:.4f}--------------"
            .format(fold + 1, acc * 100, f1, mse, mis))
