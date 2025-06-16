import numpy as np
import torch
from torch.utils.data import random_split


def get_accuracy(logits, y):
    pred_label = torch.argmax(logits, dim=-1)
    return torch.sum(pred_label == y) / len(pred_label)


def dataset_split(dataset, split_ratio=0.95):
    data_len = len(dataset)

    train_len = int(data_len * split_ratio)
    valid_len = data_len - train_len

    train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])

    return train_dataset, valid_dataset


# Mixup is referred to https://github.com/facebookresearch/mixup-cifar10

def mixup_data(x_wavs, x_mels, y, device, alpha=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda,
    x_wav(32,1,160000),
    x_mel(32,1,128,313),
    y_a(32)'''

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    # lam = max(lam, 1 - lam)
    batch_size = x_mels.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x_wavs = lam * x_wavs + (1 - lam) * x_wavs[index, :]
    mixed_x_mels = lam * x_mels + (1 - lam) * x_mels[index, :]

    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed_x_wavs, mixed_x_mels, y_a, y_b, lam, mixed_y


def os_scl(criterion, pred, y_a, y_b, lam, sup, feature):
    loss1 = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    loss2 = sup(feature.unsqueeze(1), y_a)
    loss = loss2 + loss1
    return loss
