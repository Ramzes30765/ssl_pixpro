from torch.utils.data import DataLoader
import torch
import numpy as np


def compute_mean_std(dataset, batch_size=32, num_workers=4):
    """
    Вычисляет среднее и стандартное отклонение по каналам для датасета.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    mean = 0.0
    std = 0.0
    nb_samples = 0
    for data, _ in loader:
        batch_samples = data.size(0)
        # [B, C, H, W] -> [B, C, H*W]
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    return mean.numpy().tolist(), std.numpy().tolist()