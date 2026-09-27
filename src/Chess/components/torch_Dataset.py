import os.path

import torch
from torch.utils.data import Dataset
import numpy as np
from Chess.config.configuration import PrepareTorchDataset


class ChessDataset(Dataset):
    def __init__(self , config : PrepareTorchDataset):
        self.config = config
        self.x  = np.load(self.config.x_data_path)
        self.y = np.load(self.config.y_data_path)

    def __len__(self):
        return len(self.x)


    def __getitem__(self, idx):
        y_idx = torch.tensor(self.y[idx]).float()
        x_idx = torch.from_numpy(self.x[idx]).float()
        return  x_idx ,  y_idx,

    def save_data(self):
        x_tensor = torch.from_numpy(self.x).float()
        y_tensor = torch.tensor(self.y).float()

        x_path = os.path.join(self.config.root_dir , "x_tensor.pt")
        y_path = os.path.join(self.config.root_dir , "y_tensor.pt")

        torch.save(x_tensor , x_path)
        torch.save(y_tensor , y_path)

import torch

x = torch.load("artifacts/torch_dataset/x_tensor.pt")
print(type(x))
print(x.shape)
print(x.dtype)