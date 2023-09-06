'''
domain B에 대한 프로토타입 생성을 위한 target class에 대한 소량의 데이터셋 생성이 목적
두가지 테스트가 필요함 
1. SVL 방식으로 학습된 classifier를 이용해서 domain B에 대한 학습 진행 후 fake데이터 분류 테스트
2. fake데이터로 domain B 프로토타입 생성 후 실제 데이터로 분류해보기
'''

# SVL Test
import os
import tqdm
import torch
import torch.utils.data as DATA
import torch.nn.functional as F
import numpy as np
from dataloader.dataset import SVLDataset, FakeDataset
from models.vit import ViT
from runner.utils import torch_seed, get_config
from torch.optim import lr_scheduler, Adam

# fix torch seed
torch_seed(40)

cfg = get_config('config.yaml')
batch_size = 64


# Load Dataset
train_data = SVLDataset(cfg['test_dataset_path'], win_size=cfg["window_size"], mode='train', train_proportion=0.8)
test_data = SVLDataset(cfg['test_dataset_path'], win_size=cfg["window_size"], mode='test', train_proportion=0.8)
fake_data = FakeDataset(data_path=cfg['fake_path'], win_size=64)

train_dataloader = DATA.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader =  DATA.DataLoader(test_data, batch_size=batch_size, shuffle=True)
fake_dataloader =  DATA.DataLoader(fake_data, batch_size=batch_size, shuffle=True)

model = ViT(
        in_channels=cfg["in_channels"],
        patch_size=(cfg["patch_size"], cfg[cfg["bandwidth"]]),
        embed_dim=cfg["embed_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        mlp_dim=cfg["mlp_dim"],
        num_classes=len(cfg["activity_labels"]),
        in_size=[cfg["window_size"], cfg[cfg["bandwidth"]]]
    ).cuda(0)


epoch = 50
criterion = torch.nn.CrossEntropyLoss()
optimizer =  torch.optim.Adam(model.parameters(), lr=cfg['lr'])
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

model.train()
for e in range(epoch):
    print('Epoch {}/{}'.format(e + 1, epoch))
    print('-' * 10)
    
    train_loss = 0.0
    train_acc = 0.0
    total_iter = 0
    
    for i, data in enumerate(tqdm.tqdm(train_dataloader)):
        data_x, data_y = data
        data_x = data_x.unsqueeze(1).float().cuda(0)
        data_y = data_y.cuda(0)

        optimizer.zero_grad()
        
        outputs = model(data_x)
        loss = criterion(outputs, data_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        # Calculate accuracy
        outputs = F.log_softmax(outputs, dim=0)
        y_hat = torch.from_numpy(np.array([np.argmax(outputs.cpu().data.numpy()[i]) for i in range(len(outputs))]))
        data_y = torch.from_numpy(np.array([np.argmax(data_y.cpu().data.numpy()[i]) for i in range(len(data_y))]))
        train_acc += torch.eq(y_hat, data_y).float().mean()
        total_iter += 1

    epoch_loss = train_loss / total_iter
    epoch_acc = train_acc / total_iter
    print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(e + 1, epoch_loss, epoch_acc))
    scheduler.step()

    os.makedirs('./checkpoint/svl_vit', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('./checkpoint/svl_vit', "{}.tar".format(epoch)))
    print("saved at {}".format(os.path.join('./checkpoint/svl_vit', "{}.tar".format(epoch))))

# Evaluation supervised learning based ViT    
model.eval()

test_acc = 0
test_loss = 0
total_iter = 0

with torch.no_grad():
    for i, data in enumerate(tqdm.tqdm(test_dataloader)):
        data_x, data_y = data
        data_x = data_x.unsqueeze(1).float().cuda(0)
        data_y = data_y.cuda(0)

        outputs = model(data_x)
        loss = criterion(outputs, data_y)
        test_loss += loss.item()
        
        # Calculate accuracy
        outputs = F.log_softmax(outputs, dim=0)
        y_hat = torch.from_numpy(np.array([np.argmax(outputs.cpu().data.numpy()[i]) for i in range(len(outputs))]))
        data_y = torch.from_numpy(np.array([np.argmax(data_y.cpu().data.numpy()[i]) for i in range(len(data_y))]))
        test_acc += torch.eq(y_hat, data_y).float().mean()
        total_iter += 1
    
    test_loss = test_loss / total_iter
    test_acc = test_acc / total_iter
    print('Test Result -- Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))


fake_acc = 0
fake_loss = 0
total_iter = 0
with torch.no_grad():
    for i, data in enumerate(tqdm.tqdm(fake_dataloader)):
        data_x, data_y = data
        data_x = data_x.unsqueeze(1).float().cuda(0)
        data_y = data_y.cuda(0)

        outputs = model(data_x)
        loss = criterion(outputs, data_y)
        fake_loss += loss.item()
        
        # Calculate accuracy
        outputs = F.log_softmax(outputs, dim=0)
        y_hat = torch.from_numpy(np.array([np.argmax(outputs.cpu().data.numpy()[i]) for i in range(len(outputs))]))
        data_y = torch.from_numpy(np.array([np.argmax(data_y.cpu().data.numpy()[i]) for i in range(len(data_y))]))
        fake_acc += torch.eq(y_hat, data_y).float().mean()
        total_iter += 1
    
    fake_loss = fake_loss / total_iter
    fake_acc = fake_acc / total_iter
    print('Fake Test Result -- Loss: {:.4f} Acc: {:.4f}'.format(fake_loss, fake_acc))



