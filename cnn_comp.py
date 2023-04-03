import pandas as pd
import numpy as np
from scipy.io import loadmat
import multiprocessing as mp
import os
import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def read_mat(csi_directory_path, csi_action):
    """
    Reads all the actions from a given alphabet_directory
    """
    datax = []
    datay = []

    csi_mats = os.listdir(csi_directory_path)
    for csi_mat in csi_mats:
        mat = loadmat(csi_directory_path + csi_mat)
        if 'PCA' in csi_directory_path:
            data = mat['cfm_data']
        else:
            data = mat['iq_data']

        datax.extend([data])
        datay.extend([csi_action])
    return np.array(datax), np.array(datay)

def read_csi(base_directory):
    """
    Reads all the alphabets from the base_directory
    Uses multithreading to decrease the reading time drasticallytrain_x
    """
    datax = None
    datay = None
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply(read_mat,args=(
                              base_directory + '/' + directory + '/', directory, 
                              )) for directory in os.listdir(base_directory)]
    pool.close()
    for result in results:
        if datax is None:
            datax = result[0]
            datay = result[1]
        else:
            datax = np.vstack([datax, result[0]])
            datay = np.concatenate([datay, result[1]])
    return datax, datay


data_folder = 'extractd_3x4/m1c4_PCA_80_300_extracted_3x4'
train_folder_name = 'few_shot_datasets/' + data_folder + '/train_A1'

model_out_name = 'models/model_' + data_folder + '.pt'


trainx, trainy = read_csi(train_folder_name)
trainx = np.expand_dims(trainx, axis=1)


for i in range(trainy.shape[0]):
    if trainy[i]=='walk':
        trainy[i] = 0
    elif trainy[i]=='empty':
        trainy[i] = 1
    elif trainy[i] == 'jump':
        trainy[i] = 2
    else:
        trainy[i] = 3
# no train test
# tain_x = trainx
# train_y = trainy
 
train_x, test_x, train_y, test_y = train_test_split(trainx, trainy, test_size = 0.1, random_state=40, shuffle=True)


datax  = torch.from_numpy(train_x).float()

# converting the target into torch format
train_y = train_y.astype(int);
datay = torch.from_numpy(train_y)


# converting validation images into torch format
test_x  = torch.from_numpy(test_x).float()

# converting the target into torch format
test_y = test_y.astype(int);
test_y = torch.from_numpy(test_y)


#%%
class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(16 * 16 * 64, 1000),
            ReLU(inplace=True),
            Linear(1000, 80),
            ReLU(inplace=True),
            Linear(80, 40),
            ReLU(inplace=True),
            Linear(40, 4)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.001)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    
print(model)

def extract_batch(datax, datay, batch_size = 5):
    num_batch = torch.floor(torch.tensor(datax.shape[0]/batch_size)).int()
    batch_x = torch.zeros(num_batch, batch_size, datax.shape[1],datax.shape[2],datax.shape[3])
    batch_y = torch.zeros(num_batch, batch_size)
    for i in range(num_batch):
        x = datax[i*batch_size:(i+1)*batch_size,:,:,:]
        y = datay[i*batch_size:(i+1)*batch_size]
        batch_x[i,:,:,:,:] = x
        batch_y[i,:] = y
    return (num_batch, batch_x, batch_y)


def train(epoch):
    model.train()

    for i in range(num_batch):
        tr_loss = 0
        x_t = torch.squeeze(batch_x[i,:,:,:,:],0)
        y_t = torch.squeeze(batch_y[i,:],0).long()
        # getting the training set
        x_t, y_t = Variable(x_t), Variable(y_t)
        x_t = x_t.cuda()
        y_t = y_t.cuda()
        # getting the validation set
        optimizer.zero_grad()
        output_train = model(x_t)
        loss_train = criterion(output_train, y_t)
        train_losses.append(loss_train)
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()      
    if epoch%100 == 0:
        # printing the validation loss
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_train)

# %%

n_epochs = 600
# empty list to store training losses
train_losses = []
num_batch, batch_x, batch_y = extract_batch(datax, datay, batch_size = 5)
# num_batch_v, batch_x_v, batch_y_v = extract_batch(x_val, y_val, batch_size = 5)
# empty list to store validation losses
# training the model
for epoch in range(n_epochs):
    train(epoch)
# %%
with torch.no_grad():
    output = model(datax.cuda())
    
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on training set
print('Training acc: ', accuracy_score(datay, predictions))

with torch.no_grad():
    output = model(test_x.cuda())

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
print('Testing acc: ',accuracy_score(test_y, predictions))
print('CF: ', confusion_matrix(test_y, predictions, normalize='true'))
# %%
test_folder_name = 'few_shot_datasets/' + data_folder + '/test_A2'
testx, testy = read_csi(test_folder_name)
testx = np.expand_dims(testx, axis=1)
test_x  = torch.from_numpy(testx).float() 

result_out_name = 'results/result_A1A3_' + data_folder + '.pt'

for i in range(testy.shape[0]):
    if testy[i]=='walk':
        testy[i] = 0
    elif testy[i]=='empty':
        testy[i] = 1
    elif testy[i] == 'jump':
        testy[i] = 2
    else:
        testy[i] = 3
test_y = testy.astype(int);
test_y = torch.from_numpy(test_y)
   
# %%
with torch.no_grad():
    output = model(test_x.cuda())

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
print('Testing acc: ',accuracy_score(test_y, predictions))
print('CF: ', confusion_matrix(test_y, predictions, normalize='true'))
