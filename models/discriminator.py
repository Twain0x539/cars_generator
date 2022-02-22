import torch
from torch import nn


class PatchDicriminator(nn.Module):

    def __init__(self, in_channels=6, nf=64):
        super(PatchDicriminator, self).__init__()
        self.relu = nn.LeakyReLU(0.2, True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.bn = nn.BatchNorm2d(nf)
        self.bn2 = nn.BatchNorm2d(nf * 2)
        self.bn4 = nn.BatchNorm2d(nf * 4)
        self.bn8 = nn.BatchNorm2d(nf * 8)
        self.bn16 = nn.BatchNorm2d(nf * 16)

        self.conv1 = nn.Conv2d(in_channels, nf, 4, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf * 2, 4, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv2d(nf * 2, nf * 4, 4, stride=2, padding=1, bias=True)
        self.conv4 = nn.Conv2d(nf * 4, nf * 8, 4, stride=2, padding=1, bias=True)
        self.conv5 = nn.Conv2d(nf * 8, nf * 16, 4, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(nf * 16, 1, 4, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn4(self.conv3(x)))
        x = self.relu(self.bn8(self.conv4(x)))
        x = self.relu(self.bn16(self.conv5(x)))
        x = self.sigmoid(self.conv6(x))

        return x

