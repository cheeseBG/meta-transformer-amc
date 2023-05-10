import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 2), padding='same')
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 2), padding='same')
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 2), padding='same')
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 2), padding='same')
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0)

    def forward(self, x, max_pool):
        identity = self.conv1(x)

        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x += identity
        x = F.relu(x)

        identity = x

        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x += identity
        x = F.relu(x)

        if max_pool:
            x = self.maxpool(x)

        return x


class ResNetStack(nn.Module):
    def __init__(self, in_channels, n_class):
        super(ResNetStack, self).__init__()

        self.reshape = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock(1, 32, "ReStk1"),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding='valid'),
            ResidualBlock(32, 32, "ReStk2"),
            ResidualBlock(32, 32, "ReStk3"),
            ResidualBlock(32, 32, "ReStk4"),
            ResidualBlock(32, 32, "ReStk5"),
            ResidualBlock(32, 32, "ReStk6")
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.SELU(),
            nn.AlphaDropout(0.3),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.AlphaDropout(0.3),
            nn.Linear(128, n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.reshape(x)
        x = self.residual_blocks(x)
        x = self.fc_layers(x)

        return x
