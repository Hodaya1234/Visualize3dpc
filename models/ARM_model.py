import torch
import torch.nn as nn


class ARMLayer(nn.Module):
    def __init__(self, classes=40):
        super().__init__()
        self.classes = classes
        self.relu = nn.ReLU()

        self.linear3 = nn.Linear(512, self.classes)


    def forward(self, x, train=True):
        # x: B x 1024 x Ns
        x = x.permute(0, 2, 1)  # x: B x Ns x 512
        ARM = self.linear3(x)
        out = torch.mean(ARM, dim=1)# out: B  x 40

        if train:
            return out
        else:
            return ARM, out
