import torch
import torch.nn as nn

class SimpleRNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleRNNClassifier, self).__init__()
        
        # RNN 레이어 정의
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        # 소프트맥스 레이어 정의
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # RNN 레이어를 통과한 후, 마지막 타임 스텝의 출력을 가져옵니다.
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # 마지막 타임 스텝의 출력
        
        # 소프트맥스 레이어를 통과하여 클래스별 확률을 얻습니다.
        out = self.fc(out)
        
        return out

# 모델 인스턴스 생성
input_size = 1024
hidden_size = 128  # 임의의 은닉 상태 크기를 선택할 수 있습니다.
num_classes = 24
model = SimpleRNNClassifier(input_size, hidden_size, num_classes)

# 모델 정보 출력
print(model)
