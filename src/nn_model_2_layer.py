import torch.nn as nn


class TwoLayerNet(nn.Module):
    """
    Two hidden layer MLP with BatchNorm and Dropout.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)