import numpy as np

import torch
from torch.utils import data


def augment_points(x: np.ndarray):
    if np.random.rand() < 0.5:
        x += np.random.normal(0, 0.02, size=x.shape)
    if np.random.rand() < 0.5:
        theta = np.random.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        x = x @ rot.T
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.9, 1.1)
        x *= scale
    if np.random.rand() < 0.5:
        trans = np.random.uniform(-0.1, 0.1, size=(1, 2))
        x += trans
    return x.astype(np.float32)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x_path, y_path, scaler=None, augment=False):
        self.X = np.load(x_path)
        self.y = np.load(y_path)
        self.augment = augment

        if scaler is not None:
            N, P, D = self.X.shape
            X_flat = self.X.reshape(-1, D)
            X_scaled = scaler.transform(X_flat)
            self.X = X_scaled.reshape(N, P, D)
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index].copy()
        y = self.y[index]
        if self.augment:
            x = augment_points(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)