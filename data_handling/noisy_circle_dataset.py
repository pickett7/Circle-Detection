import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class NoisyImages(Dataset):
    def __init__(self, path_to_trainset, transform=None):
        self.dataset = pd.read_csv(path_to_trainset, sep=' ')
        self.transform = transform

    def __getitem__(self, idx):
        image = np.load(self.dataset.iloc[idx, 0].split(',')[0])
        target = [self.dataset.iloc[idx, 0].split(',')[i] for i in range(1, 4)]
        
        if self.transform:
            image = np.expand_dims(np.asarray(image), axis=0)
            image = torch.from_numpy(np.array(image, dtype=np.float32))
            target = torch.from_numpy(np.array(np.asarray(target), dtype=np.float32))
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.dataset)