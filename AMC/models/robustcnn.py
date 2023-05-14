import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from thop import profile
import time


class ABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((1, 2)),

            nn.Conv2d(16, 16, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((1, 2))
        )

    def forward(self, x):
        return self.layer(x)


class BBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x + self.layer2(x)
        return x


class CBlock1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 24, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 64, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x + self.layer2(x)
        return x


class CBlock2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x + self.layer2(x)
        return x



class RobustCNN(nn.Module):
    def __init__(self, n_class=24, softmax=True):
        super().__init__()
        self.softmax = softmax
        self.n_class = n_class
        self.a = ABlock()
        self.b = BBlock()
        self.c1 = CBlock1()
        self.c2 = CBlock2()

        self.avgpool = nn.AvgPool2d(2, 1)
        self.aavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128*1*1, self.n_class)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.avgpool(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.aavgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.softmax:
            x = F.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    model = RobustCNN().to("cuda")
    print(summary(model, (1, 4, 1024)))

    input = torch.randn(1, 1, 4, 1024).cuda()

    start_time = time.time()
    outputs = model(input)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(
        "Elapsed time: %.3f" % (elapsed_time)
    )

    input = torch.randn(1, 1, 4, 1024)

    macs, params = profile(model, inputs=(torch.Tensor(input).to(device="cuda"),))
    print(
        "Param: %.2fM | FLOPs: %.3fG" % (params / (1000 ** 2), macs / (1000 ** 3))
    )
