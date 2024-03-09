import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 3)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(3, 2)

    def forward(self, X):
        out = self.fc1(X)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.out(out)
        return out
