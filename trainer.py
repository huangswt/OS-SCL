import json
import os

import torch

from tqdm import tqdm
from utils import get_accuracy, mixup_data, os_scl
from losses import ASDLoss, SupConLoss, WeightEMA


class Trainer:
    def __init__(self, device, net, ema_net, epochs=300, lr=0.0001, cfg=None):
        self.device = device
        self.epochs = epochs
        self.net = net
        self.ema_net = ema_net
        self.cfg = cfg

        self.suploss = SupConLoss()

        model_name = type(self.net).__name__

        if cfg["desc"] != 'None':
            model_name += f'_{cfg["desc"]}'
        self.cfg['model_name'] = model_name
        os.makedirs(f'./check_points/{model_name}', exist_ok=True)

        self.model_path = f'./check_points/{model_name}/model.pth'
        with open(
                f'./check_points/{model_name}/config.json',
                'w',
                encoding='utf-8') as json_file:
            json.dump(cfg, json_file, ensure_ascii=False, indent=4)

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)

        self.ema_optimizer = WeightEMA(self.net, self.ema_net)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs,
                                                                    eta_min=0.1 * float(lr))
        self.criterion = ASDLoss().to(self.device)
        self.test_criterion = ASDLoss(reduction=False).to(self.device)

    def train(self, train_loader, valid_loader):
        num_steps = len(train_loader)
        min_val_loss = 1e10

        for epoch in tqdm(range(self.epochs), total=self.epochs, dynamic_ncols=True):
            sum_loss = 0.
            sum_accuracy = 0.

            for _, (x_wavs, x_mels, labels) in tqdm(enumerate(train_loader), total=num_steps, dynamic_ncols=True):
                self.net.train()

                x_wavs, x_mels, labels = x_wavs.to(self.device), x_mels.to(self.device), labels.to(self.device)

                mixed_x_wavs, mixed_x_mels, y_a, y_b, lam, mixed_y = mixup_data(x_wavs, x_mels, labels,
                                                                                self.device)
                logits, feature = self.net(mixed_x_wavs, mixed_x_mels, labels)
                loss = os_scl(self.criterion, logits, y_a, y_b, lam,
                              self.suploss,
                              feature)

                sum_accuracy += get_accuracy(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_optimizer.step()

                sum_loss += loss.item()
            self.scheduler.step()

            avg_loss = sum_loss / num_steps
            avg_accuracy = sum_accuracy / num_steps

            valid_loss, valid_accuracy = self.valid(valid_loader)
            if min_val_loss > valid_loss:
                min_val_loss = valid_loss
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f'lr: {lr:.7f} | EPOCH: {epoch} | Train_loss: {avg_loss:.5f} | Train_accuracy: {avg_accuracy:.5f} | Valid_loss: {valid_loss:.5f} | Valid_accuracy: {valid_accuracy:.5f}')
                torch.save(self.ema_net.state_dict(), self.model_path)

    @torch.no_grad()
    def valid(self, valid_loader):

        net = self.ema_net.eval()

        num_steps = len(valid_loader)
        sum_loss = 0.
        sum_accuracy = 0.

        for (x_wavs, x_mels, labels) in valid_loader:
            x_wavs, x_mels, labels = x_wavs.to(self.device), x_mels.to(self.device), labels.to(self.device)

            logits, _ = net(x_wavs, x_mels, labels, train=True)

            sum_accuracy += get_accuracy(logits, labels)
            loss = self.criterion(logits, labels)

            sum_loss += loss.item()

        avg_loss = sum_loss / num_steps
        avg_accuracy = sum_accuracy / num_steps
        return avg_loss, avg_accuracy
