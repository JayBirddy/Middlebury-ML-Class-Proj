import torch.nn as nn

class OneLayerNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.pipeline(x)