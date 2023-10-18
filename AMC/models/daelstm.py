import torch.nn as nn
import torch.nn.functional as F

class DAELSTM(nn.Module):
    def __init__(self, input_shape, modulation_num):
        super(DAELSTM, self).__init__()
        
        self.encoder_1 = nn.LSTM(input_size=input_shape[1], 
                                 hidden_size=32, 
                                 num_layers=1, 
                                 batch_first=True)
        
        self.drop_1 = nn.Dropout(p=0)
        
        self.encoder_2 = nn.LSTM(input_size=32, 
                                 hidden_size=32, 
                                 num_layers=1, 
                                 batch_first=True)
        
        self.decoder = nn.Linear(32, 2)
        
        self.clf_dense_1 = nn.Linear(32, 32)
        self.bn_1 = nn.BatchNorm1d(32)
        self.clf_drop_1 = nn.Dropout(p=0)
        
        self.clf_dense_2 = nn.Linear(32, 16)
        self.bn_2 = nn.BatchNorm1d(16)
        self.clf_drop_2 = nn.Dropout(p=0)
        
        self.clf_dense_3 = nn.Linear(16, modulation_num)
        
    def forward(self, x):
        encoder_output_1, (state_h_1, state_c_1) = self.encoder_1(x)
        drop_1_out = self.drop_1(encoder_output_1)
        
        encoder_output_2, (state_h_2, state_c_2) = self.encoder_2(drop_1_out)
        decoder_out = self.decoder(encoder_output_2)
        
        x = F.relu(self.clf_dense_1(state_h_2.squeeze(0)))
        x = self.clf_drop_1(self.bn_1(x))
        x = F.relu(self.clf_dense_2(x))
        x = self.clf_drop_2(self.bn_2(x))
        x = self.clf_dense_3(x)
        
        return x

if __name__ == '__main__':
    import torch
    import time
    from pytorch_model_summary import summary as summary
    from thop import profile

    input = torch.randn(1, 1024, 2).to("cuda")
    model = DAELSTM(input_shape=[1,2,1024],
                   modulation_num=11).to("cuda")
    print(summary(model, input))

    input = torch.randn(2, 1024, 2).cuda()
    start_time = time.time()
    outputs = model(input)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(
        "Elapsed time: %.3f" % (elapsed_time)
    )

    macs, params = profile(model, inputs=(input.to(device="cuda"),))
    print(
        "Param: %.2fM | FLOPs: %.3fG" % (params / (1000 ** 2), macs / (1000 ** 3))
    )
    total_params = sum(p.numel() for p in model.parameters())

    # 각 파라미터의 데이터 타입에 따른 크기 계산
    total_size_bytes = total_params * 4  # 32비트(float32)의 경우 4바이트 사용

    # 바이트를 메가바이트로 변환
    total_size_megabytes = total_size_bytes / (1024 * 1024)

    print(f"모델 예상 총 크기: {total_size_megabytes:.2f} MB")