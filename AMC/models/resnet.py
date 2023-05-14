class ResidualUnit(nn.Module):
    def __init__(self, in_channels, max_pool=False):
        super(ResidualUnit, self).__init__()
        # 1*1 Conv
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        # Residual Unit 1
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(2, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=(2, 3), padding=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=(2, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=(2, 3), padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

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
        out = self.relu(self.conv4(x))
        out = self.conv5(out)
        out += identity
        out = self.relu(out)

        if self.max_pool:
            out = self.max_pool(out)

        return out


class ResNetStack(nn.Module):
    def __init__(self):
        super(ResNetStack, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 1))
        self.res_unit1 = ResidualUnit(32, False)
        self.res_units = nn.ModuleList([ResidualUnit(32, True) for _ in range(5)])
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2))
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 24)
        self.drop = nn.AlphaDropout(0.3)

    def forward(self, x):
        x = self.res_unit1(x)
        x = self.max_pool(x)

        for res_unit in self.res_units:
            x = res_unit(x)

        x = torch.flatten(x, 1)
        x = F.selu(self.fc1(x))
        x = self.drop(x)
        x = F.selu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)

        return x

# class residual_unit(torch.nn.Module):
#     def __init__(self, training=False):
#         super(residual_unit, self).__init__()
#         self.ru_conv1 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=True)
#         self.ru_bn1 = torch.nn.BatchNorm1d(32, affine=training)
#         self.ru_act1 = torch.nn.ReLU()
#         self.ru_conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=True)
#         self.ru_bn2 = torch.nn.BatchNorm1d(32, affine=training)
#         self.ru_act2 = torch.nn.ReLU()
#
#     def forward(self, x):
#         y = self.ru_conv1(x)
#         y = self.ru_bn1(y)
#         y = self.ru_act1(y)
#         y = self.ru_conv2(y)
#         y = self.ru_bn2(y)
#         y = y + x
#         y = self.ru_act2(y)
#         return y
#
#
# class residual_stack(torch.nn.Module):
#     def __init__(self, in_channels, training=False):
#         super(residual_stack, self).__init__()
#         self.rs_conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=1, bias=False)
#         self.rs_bn1 = torch.nn.BatchNorm1d(32, affine=training)
#         self.rs_ru1 = residual_unit(training)  # Create an object of the custom nn model
#         self.rs_ru2 = residual_unit(training)
#         self.rs_mp1 = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#
#     def forward(self, x):
#         y = self.rs_conv1(x)
#         y = self.rs_bn1(y)
#         y = self.rs_ru1(y)
#         y = self.rs_ru2(y)
#         y = self.rs_mp1(y)
#         return y
#
#
# class ResNetStack(torch.nn.Module):
#     def __init__(self, training=False):
#         super(ResNetStack, self).__init__()
#         self.rn33_rs1 = residual_stack(2)  # output is N*32*512
#         self.rn33_rs2 = residual_stack(32)  # output is N*32*256
#         self.rn33_rs3 = residual_stack(32)  # output is N*32*128
#         self.rn33_rs4 = residual_stack(32)  # output is N*32*64
#         self.rn33_rs5 = residual_stack(32)  # output is N*32*32
#         self.rn33_rs6 = residual_stack(32)  # output is N*32*16
#         self.flat = torch.nn.Flatten()  # output is N*512
#         self.fc1 = torch.nn.Linear(512, 128)  # output is N*128
#         self.selu1 = torch.nn.SELU()
#         self.alphadrop1 = torch.nn.AlphaDropout(p=0.95)
#         self.fc2 = torch.nn.Linear(128, 128)  # output is N*128
#         self.selu2 = torch.nn.SELU()
#         self.alphadrop2 = torch.nn.AlphaDropout(p=0.95)
#         self.fc3 = torch.nn.Linear(128, 24)  # output is N*24
#         self.smx1 = torch.nn.Softmax()  # dimension
#
#     def forward(self, x):
#         y = self.rn33_rs1(x)
#         y = self.rn33_rs2(y)
#         y = self.rn33_rs3(y)
#         y = self.rn33_rs4(y)
#         y = self.rn33_rs5(y)
#         y = self.rn33_rs6(y)
#
#         # 85272 parameters
#         y = self.flat(y)
#         y = self.fc1(y)
#         y = self.selu1(y)
#         y = self.alphadrop1(y)
#
#         y = self.fc2(y)
#         y = self.selu2(y)
#         y = self.alphadrop2(y)
#
#         y = self.fc3(y)
#
#         return y


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, seq):
#         super(ResidualBlock, self).__init__()
#
#         # conv1
#         kernel_size1 = (1, 1)
#         padding1 = (0, 0)
#
#         # conv2, conv3, conv4, conv5
#         kernel_size2 = (2, 3)
#         padding2 = ((kernel_size2[0] - 1) // 2, (kernel_size2[1] - 1) // 2)
#
#         # maxpool
#         kernel_size_maxpool = (2, 1)
#         stride_maxpool = (2, 1)
#         padding_maxpool = (0, 0)
#
#         # 레이어 정의
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size1, padding=padding1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size2, padding=padding2)
#         self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size2, padding=padding2)
#         self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size2, padding=padding2)
#         self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size2, padding=padding2)
#         self.maxpool = nn.MaxPool2d(kernel_size=kernel_size_maxpool, stride=stride_maxpool, padding=padding_maxpool)
#
#     def forward(self, x, max_pool=False):
#         identity = self.conv1(x)
#         print(identity.shape)
#
#         x = F.relu(self.conv2(identity))
#         print(x.shape)
#         x = self.conv3(x)
#         print(x.shape)
#         exit()
#         x += identity
#         x = F.relu(x)
#
#         identity = x
#
#         x = F.relu(self.conv4(x))
#         x = self.conv5(x)
#         x += identity
#         x = F.relu(x)
#
#         if max_pool:
#             x = self.maxpool(x)
#
#         return x
#
#
# class ResNetStack(nn.Module):
#     def __init__(self, in_channels, n_class, softmax=True):
#         super(ResNetStack, self).__init__()
#
#         self.softmax = softmax
#
#         self.residual_blocks = nn.Sequential(
#             ResidualBlock(1, 32, "ReStk1"),
#             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=0),
#             ResidualBlock(32, 32, "ReStk2"),
#             ResidualBlock(32, 32, "ReStk3"),
#             ResidualBlock(32, 32, "ReStk4"),
#             ResidualBlock(32, 32, "ReStk5"),
#             ResidualBlock(32, 32, "ReStk6")
#         )
#
#         self.fc_layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(512, 128),
#             nn.SELU(),
#             nn.AlphaDropout(0.3),
#             nn.Linear(128, 128),
#             nn.SELU(),
#             nn.AlphaDropout(0.3),
#             nn.Linear(128, n_class)
#         )
#
#     def forward(self, x):
#         x = self.residual_blocks(x)
#         x = self.fc_layers(x)
#
#         if self.softmax:
#             x = F.softmax(x, dim=1)
#         return x