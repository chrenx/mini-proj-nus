import torch
import numpy as np


def row_normalize(v):
    if isinstance(v, torch.Tensor):
        mu = torch.mean(v, dim=1, keepdim=True)
        sigma = torch.std(v, dim=1, keepdim=True)
        return (v - mu) / sigma
    else:
        mu = np.mean(v, axis=1)
        sigma = np.std(v, axis=1)
        return (v - mu[:, None]) / sigma[:, None]
