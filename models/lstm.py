import torch
import torch.nn as nn
import numpy as np

# LSTM input (Batch_size, Sequence, Input)
#                          1024     2
# output    (Batch_size, sequence, hidden_size)
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.drop1 = nn.Dropout(inplace=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.drop2 = nn.Dropout(inplace=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.lstm1(x)
        x = self.drop1(x)
        x = self.lstm2(x)
        x = self.drop2(x)
        x = self.fc(x)
        
        return x

# # 모델 인스턴스 생성
# input_size = 1024
# hidden_size = 128  # 임의의 은닉 상태 크기를 선택할 수 있습니다.
# num_classes = 24
# model = LSTM(input_size, hidden_size, num_classes)

# # 모델 정보 출력
# print(model)
