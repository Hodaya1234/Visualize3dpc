import torch
import torch.nn as nn


class CAMLayer(nn.Module):
    def __init__(self, classes=40):
        super().__init__()
        self.conv1 = nn.Conv1d(512, 512, kernel_size=1, bias=False) #Delete
        self.conv2 = nn.Conv1d(512, 256, kernel_size=1, bias=False)
        self.fc = nn.Linear(256, classes)

    def forward(self, x, train=True):
        batch_size, _, _ = x.size()
        x = self.conv2(x)
        F=x.clone()
        x = torch.mean(x, 2)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        if train:
            return x
        else:
            return F,x

