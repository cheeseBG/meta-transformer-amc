import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from thop import profile
import time


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ProtoNet_CNN(nn.Module):
    def __init__(self, in_channels, hid_dim, z_dim):
        super(ProtoNet_CNN, self).__init__()

        self.in_channels = in_channels
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        self.conv_block_first = Conv_block(in_channels, hid_dim)
        self.conv_blocks = nn.ModuleList([Conv_block(hid_dim, hid_dim) for _ in range(6)])
        self.conv_block_last = Conv_block(hid_dim, z_dim)
        self.flatten = Flatten()

    def forward(self, x):
        x = self.conv_block_first(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.conv_block_last(x)
        x = self.flatten(x)

        return x


if __name__ == '__main__':
    model = ProtoNet_CNN(1, 32, 24).to("cuda")
    print(summary(model, (1, 2, 1024)))

    input = torch.randn(1, 1, 2, 1024).cuda()

    start_time = time.time()
    outputs = model(input)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(
        "Elapsed time: %.3f" % (elapsed_time)
    )

    input = torch.randn(1, 1, 2, 1024)

    macs, params = profile(model, inputs=(torch.Tensor(input).to(device="cuda"),))
    print(
        "Param: %.2fM | FLOPs: %.3fG" % (params / (1000 ** 2), macs / (1000 ** 3))
    )

