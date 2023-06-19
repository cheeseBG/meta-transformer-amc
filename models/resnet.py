import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from thop import profile
import time


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=False):
        super(ResidualUnit, self).__init__()
        # 1*1 Conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0,1))
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0,1))
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0,1))
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0,1))
        self.relu = nn.ReLU()
        self.max_pool = max_pool

        # Glorot Uniform Initialization (also known as Xavier Uniform)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)

        # Residual unit 1
        identity = x
        out = self.relu(self.conv2(x))
        out = self.conv3(out)
        out += identity
        out = self.relu(out)

        # Residual unit 2
        identity = out
        out = self.relu(self.conv4(out))
        out = self.conv5(out)
        out += identity
        out = self.relu(out)

        if self.max_pool:
            out = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))(out)

        return out


class ResNetStack(nn.Module):
    def __init__(self):
        super(ResNetStack, self).__init__()
        self.res_unit1 = ResidualUnit(2, 32, True)
        self.res_units = nn.ModuleList([ResidualUnit(32, 32, True) for _ in range(5)])
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2))
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 24)
        self.drop = nn.AlphaDropout(0.3)

    def forward(self, x):
        x = self.res_unit1(x)
        #x = self.max_pool(x)
        for res_unit in self.res_units:
            x = res_unit(x)
        x = torch.flatten(x, 1)
        x = F.selu(self.fc1(x))
        x = self.drop(x)
        x = F.selu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    model = ResNetStack().to("cuda")
    print(summary(model, (2, 1, 1024)))

    input = torch.randn(1, 2, 1, 1024).cuda()

    start_time = time.time()
    outputs = model(input)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(
        "Elapsed time: %.3f" % (elapsed_time)
    )

    input = torch.randn(1, 2, 1, 1024)

    macs, params = profile(model, inputs=(torch.Tensor(input).to(device="cuda"),))
    print(
        "Param: %.2fM | FLOPs: %.3fG" % (params / (1000 ** 2), macs / (1000 ** 3))
    )
