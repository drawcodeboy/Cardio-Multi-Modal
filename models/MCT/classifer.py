import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self, input_dim=32 * 3, hidden_dim=64, output_dim=7):
        super(Classifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # input_dim: token_dim * 3
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),  
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.classifier(x)
