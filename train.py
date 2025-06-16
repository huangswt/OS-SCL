import random
import torch
import numpy as np
from dataloader import train_dataset
from torch.utils.data import DataLoader
from utils import dataset_split
from trainer import Trainer
import yaml
import os
from model.net import SCLTFSTgramMFN
import argparse


def load_config(file_path):
    with open(file_path, 'r', encoding='UTF-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def save_config(cfg, file_path):
    with open(file_path, 'w', encoding='UTF-8') as f:
        yaml.dump(cfg, f)


def update_config(cfg, args):
    for key, value in vars(args).items():
        if value is not None:
            cfg[key] = value
    return cfg


def parse_arguments():
    parser = argparse.ArgumentParser(description='Modify YAML configuration.')
    parser.add_argument('--desc', type=str, default=None, help='Description to append to model name')
    parser.add_argument('--m', type=float, help='m value')
    parser.add_argument('--gpu_num', type=int, help='GPU number')
    parser.add_argument('--epoch', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--fussion', type=int, help='Fussion')
    parser.add_argument('--ht', type=str, help=' basic or leaky_relu')

    return parser.parse_args()


def main():
    args = parse_arguments()
    cfg = load_config('config.yaml')
    print('Original Configuration...')
    print(cfg)

    cfg = update_config(cfg, args)
    print('Updated Configuration...')
    print(cfg)

    random_seed(seed=2024)

    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']

    root_path = '../data/dataset'

    device = torch.device(f'cuda:{cfg["gpu_num"]}')

    print('training dataset loading...')
    dataset = train_dataset(root_path, name_list, cfg)

    train_ds, valid_ds = dataset_split(dataset)

    train_dataloader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(valid_ds, batch_size=cfg['batch_size'], num_workers=0)

    model = SCLTFSTgramMFN(num_classes=41, m=cfg['m'], cfg=cfg).to(device)

    model_ema = SCLTFSTgramMFN(num_classes=41, m=cfg['m'], cfg=cfg).to(device)

    for param in model_ema.parameters():
        param.detach_()

    trainer = Trainer(device=device, net=model, ema_net=model_ema, epochs=cfg['epoch'], cfg=cfg)

    trainer.train(train_dataloader, valid_dataloader)


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    os.makedirs('check_points', exist_ok=True)
    main()
