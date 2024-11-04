import torch

class temp_dataloader():
    def __init__(self, mode):
        self.mode = mode
    
    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        x1 = torch.randn(1000, 1)
        x2 = torch.randn(1000, 1)
        
        label = torch.tensor([idx % 7], dtype=torch.int64)
        
        return x1, x2, label