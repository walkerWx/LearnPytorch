import torch
from torch.utils.data import Dataset

class MyTrainDataset(Dataset):
    def __init__(self, size = 1000):
        self.size = size
        self.samples = [(torch.randn(28*28), torch.randn(1)) for _ in range(size)]
        return
        
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.samples[index]
    
        