import torch
from torch import nn

class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
        self.relu = nn.ReLU()
        self.li = nn.Linear(5000, 7)
    
    def forward(self, x1, x2):
        x = x1 + x2
        x = self.attn(x, x, x)[0].reshape(x.shape[0], -1)
        return self.li(self.relu(x))