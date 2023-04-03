"""
Author: Niloofar Bahadori
ReWiS for 3 monitors
Train 3 seperate models for each monitors
Training:
    1) Read csi_data and, 2)Learning the embedding through ResNet 12 and 3) ProtoNet
Testing:
    1) Read csi_data and, 2)Utilizie the learned embedding function 12 and 
    3) ProtoNet for predition, 4)Decision Fusion
    
"""
import resnet
import numpy as np
from scipy.io import loadmat
import multiprocessing as mp
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tnrange

def read_mat(csi_directory_path, csi_action):
    """
    Reads all the actions from a given action directory
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
    Reads all the data_frames from the base_directory
    Uses multithreading to decrease the reading time drastically
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
# %%
"""Load the dataset's folder:either m3c1_xxx or m3c4_xxx"""
data_folder = 'extractd_3x4/m3c4_PCA_80_300_extracted_3x4'
train_env = 'A1'
train_folder_name = 'few_shot_datasets/' + data_folder + '/train_' + train_env

out_name = 'models/output_' + data_folder + '.pt'
""" Extracting train samples from 3 monitors"""
# m1 datasets
trainx_1, trainy_1 = read_csi(train_folder_name+'/m1')
trainx_1 = np.expand_dims(trainx_1, axis=1)
# m2 datasets
trainx_2, trainy_2 = read_csi(train_folder_name+'/m2')
trainx_2 = np.expand_dims(trainx_2, axis=1)
# m3 datasets
trainx_3, trainy_3 = read_csi(train_folder_name+'/m3')
trainx_3 = np.expand_dims(trainx_3, axis=1)

model_out_name = 'models/model_' + data_folder + '.pt'

#%%
def extract_sample(n_way, n_support, n_query, datax, datay):
  """
  Picks random sample of size n_support+n_querry, for n_way classes
  Args:
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      datax (np.array): dataset of data_frames
      datay (np.array): dataset of labels
  Returns:
      (dict) of:
        (torch.Tensor): sample of data_frames. Size (n_way, n_support+n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
  """
  sample = []
  K = np.random.choice(np.unique(datay), n_way, replace=False)
  for cls in K:
    datax_cls = datax[datay == cls]
    perm = np.random.permutation(datax_cls)
    sample_cls = perm[:(n_support+n_query)]
    sample.append(sample_cls)
  sample = np.array(sample)
  sample = torch.from_numpy(sample).float()
  #sample = sample.permute(0,1,4,2,3)
  #sample = np.expand_dims(sample, axis= 0)
  return({
      'csi_mats': sample,
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query
      })


  """
  Picks random sample of size n_support+n_querry, for n_way classes
  Args:
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      datax (np.array): dataset of data_frames
      datay (np.array): dataset of labels
  Returns:
      (dict) of:
        (torch.Tensor): sample of images. Size (n_way, n_support+n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
  """
  sample = []
  K = np.random.choice(np.unique(datay), n_way, replace=False)
  for cls in K:
    datax_cls = datax[datay == cls]
    perm = np.random.permutation(datax_cls)
    sample_cls = perm[:(n_support+n_query)]
    sample.append(sample_cls)
  sample = np.array(sample)
  sample = torch.from_numpy(sample).float()
  #sample = sample.permute(0,1,4,2,3)
  #sample = np.expand_dims(sample, axis= 0)
  return({
      'csi_mats': sample,
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query
      })


#sample_example = extract_sample(2, 8, 5, trainx, trainy)

#%%

class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return x.view(x.size(0), -1)

class LinearEmbed(nn.Module):
    """Linear Embedding"""
    def __init__(self, dim_in=256*256*64, dim_out = 16*64):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

def load_protonet_conv(**kwargs):
  """
  Loads the prototypical network model
  Arg:
      x_dim (tuple): dimension of input data
      hid_dim (int): dimension of hidden layers in conv blocks
      z_dim (int): dimension of embedded data
  Returns:
      Model (Class ProtoNet)
  """
  x_dim = kwargs['x_dim']
  hid_dim = kwargs['hid_dim']
  z_dim = kwargs['z_dim']
  
  

  def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
        )
    
  encoder = nn.Sequential(
    conv_block(x_dim[0], hid_dim),
    conv_block(hid_dim, hid_dim),
    conv_block(hid_dim, hid_dim),
    # conv_block(hid_dim, hid_dim),
    # conv_block(hid_dim, hid_dim),
    # conv_block(hid_dim, hid_dim),
    # conv_block(hid_dim, hid_dim),
    conv_block(hid_dim, z_dim),
    Flatten()
    )
    
  return ProtoNet(encoder)


class ProtoNet(nn.Module):
  def __init__(self, encoder):
    """
    Args:
        encoder : CNN encoding the images in sample
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
    """
    super(ProtoNet, self).__init__()
    self.encoder = encoder.cuda()

  def set_forward_loss(self, sample):
    """
    Computes loss, accuracy and output for classification task
    Args:
        sample (torch.Tensor): shape (n_way, n_support+n_query, (dim)) 
    Returns:
        torch.Tensor: shape(2), loss, accuracy and y_hat
    """
    sample_images = sample['csi_mats'].cuda()
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']

    x_support = sample_images[:, :n_support]
    x_query = sample_images[:, n_support:]
   
    #target indices are 0 ... n_way-1
    target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
    target_inds = Variable(target_inds, requires_grad=False)
    target_inds = target_inds.cuda()
   
    #encode images of the support and the query set
    x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                   x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])], 0)
   
    z = self.encoder.forward(x)
    z_dim = z.size(-1) #usually 64
    z_proto = z[:n_way*n_support].view(n_way, n_support, z_dim).mean(1)
    z_query = z[n_way*n_support:]

    #compute distances
    dists = euclidean_dist(z_query, z_proto)
    
    #compute probabilities
    log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
   
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
   
    return loss_val, {
        'loss': loss_val.item(),
        'acc': acc_val.item(),
        'y_hat': y_hat,
        'log_p_y': log_p_y
        }

def euclidean_dist(x, y):
  """
  Computes euclidean distance btw x and y
  Args:
      x (torch.Tensor): shape (n, d). n usually n_way*n_query
      y (torch.Tensor): shape (m, d). m usually n_way
  Returns:
      torch.Tensor: shape(n, m). For each query, the distances to each centroid
  """
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)
#%%

def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size):
  """
  Trains the protonet
  Args:
      model
      optimizer
      train_x (np.array): samples of training set
      train_y(np.array): labels of training set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      max_epoch (int): max epochs to train on
      epoch_size (int): episodes per epoch
  """
  #divide the learning rate by 2 at each epoch, as suggested in paper
  scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
  epoch = 0 #epochs done so far
  stop = False #status to know when to stop

  while epoch < max_epoch and not stop:
    running_loss = 0.0
    running_acc = 0.0

    for episode in tnrange(epoch_size, desc="Epoch {:d} train".format(epoch+1)):
      sample = extract_sample(n_way, n_support, n_query, train_x, train_y)
      optimizer.zero_grad()
      loss, output = model.set_forward_loss(sample)
      running_loss += output['loss']
      running_acc += output['acc']
      loss.backward()
      optimizer.step()
    epoch_loss = running_loss / epoch_size
    epoch_acc = running_acc / epoch_size
    print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch+1,epoch_loss, epoch_acc))
    
    epoch += 1
    scheduler.step()

# m1
model1 = load_protonet_conv(
    x_dim=(1,256,256),
    hid_dim=64,
    z_dim=64,
    )
model2 = load_protonet_conv(
    x_dim=(1,256,256),
    hid_dim=64,
    z_dim=64,
    )
model3 = load_protonet_conv(
    x_dim=(1,512,256),
    hid_dim=64,
    z_dim=64,
    )

optimizer1 = optim.Adam(model1.parameters(), lr = 0.001)
optimizer2 = optim.Adam(model2.parameters(), lr = 0.001)
optimizer3 = optim.Adam(model3.parameters(), lr = 0.001)

n_way = 4
n_support = 5
n_query = 5


max_epoch = 3
epoch_size = 1000

train(model1, optimizer1, trainx_1, trainy_1, n_way, n_support, n_query, max_epoch, epoch_size)
train(model2, optimizer2, trainx_2, trainy_2, n_way, n_support, n_query, max_epoch, epoch_size)
train(model3, optimizer3, trainx_3, trainy_3, n_way, n_support, n_query, max_epoch, epoch_size)
#%%

def test_ind(model, test_x, test_y, n_way, n_support, n_query, test_episode):
  """
  Tests the protonet
  Args:
      model: trained model
      test_x (np.array): samples of testing set
      test_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      test_episode (int): number of episodes to test on
  """
  running_loss = 0.0
  running_acc = 0.0
  for episode in tnrange(test_episode):
    sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
    loss, output = model.set_forward_loss(sample)
    running_loss += output['loss']
    running_acc += output['acc']
  avg_loss = running_loss / test_episode
  avg_acc = running_acc / test_episode
  print('Test results -- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))

#%%
def sample_test(n_way, n_support, n_query, testx_1, testy_1, testx_2, testy_2, testx_3, testy_3):
    sample_1 = []
    sample_2 = []
    sample_3 = []
    # n = np.unique(testy_1)
    # K = np.random.choice(n, n_way, replace=False)
    K = np.array(['empty', 'jump', 'stand', 'walk'])
    for cls in K:
      datax_cls_1 = testx_1[testy_1 == cls]
      datax_cls_2 = testx_2[testy_2 == cls]
      datax_cls_3 = testx_3[testy_3 == cls]
      m = min(datax_cls_1.shape[0], datax_cls_2.shape[0], datax_cls_3.shape[0])
      ind = np.random.choice(m, n_support+n_query, replace=False)
      sample_cls_1 = datax_cls_1[ind,:,:,:]
      sample_cls_2 = datax_cls_2[ind,:,:,:]
      sample_cls_3 = datax_cls_3[ind,:,:,:]
    
      sample_1.append(sample_cls_1)
      sample_2.append(sample_cls_2)
      sample_3.append(sample_cls_3)
      
    sample_1 = np.array(sample_1)
    sample_2 = np.array(sample_2)
    sample_3 = np.array(sample_3)
    sample_1 = torch.from_numpy(sample_1).float()
    sample_2 = torch.from_numpy(sample_2).float()
    sample_3 = torch.from_numpy(sample_3).float()
    
    return({'0': {'csi_mats': sample_1,
                'n_way': n_way,
                'n_support': n_support,
                'n_query': n_query},
           '1': {'csi_mats': sample_2,
                'n_way': n_way,
                'n_support': n_support,
                'n_query': n_query},
           '2': {'csi_mats': sample_3,
                'n_way': n_way,
                'n_support': n_support,
                'n_query': n_query}
           })
#%%
def test(model1, model2, model3, test_x_1, test_y_1, test_x_2, test_y_2, test_x_3, test_y_3, n_way, n_support, n_query, test_episode):
  """
  Tests the protonet
  Args:
      model: trained model
      test_x (np.array): samples of testing set
      test_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      test_episode (int): number of episodes to test on
  """
  target_inds = torch.arange(0, n_way).view(n_way, 1).expand(n_way, n_query).reshape(n_way*n_query,1)
  conf_mat = torch.zeros(n_way, n_way)
  running_acc = 0.0
  for episode in tnrange(test_episode):
    sample = sample_test(n_way, n_support, n_query, test_x_1, test_y_1, test_x_2, test_y_2, test_x_3, test_y_3)

    loss_1, output_1 = model1.set_forward_loss(sample['0'])
    loss_2, output_2 = model2.set_forward_loss(sample['1'])
    loss_3, output_3 = model3.set_forward_loss(sample['2'])
   
    out = output_1['log_p_y'] + output_2['log_p_y'] + output_3['log_p_y']
    out = out.cpu()
    true_out = out.max(2)

    for cls in range(n_way):
        conf_mat[cls,:] = conf_mat[cls,:] + torch.bincount(true_out.indices[cls,:], minlength = n_way) 
    a = true_out.indices.view(n_way* n_query, 1)    
    acc = torch.eq(a, target_inds).float().mean()
    #running_loss += output['loss']
    running_acc += acc
  #+fg+99------9`avg_loss = running_loss / test_episode
  avg_acc = running_acc / test_episode
  print('Test results -- Acc: {:.4f}'.format(avg_acc))
  return (conf_mat/(n_query*test_episode), avg_acc)
#%%
""" Extracting test samples from 3 monitors, pick either A2 or A3"""
test_env = 'A2'
test_folder_name = 'few_shot_datasets/' + data_folder + '/test_' + test_env
testx_1, testy_1 = read_csi(test_folder_name+'/m1')
testx_1 = np.expand_dims(testx_1, axis=1)

testx_2, testy_2 = read_csi(test_folder_name+'/m2')
testx_2 = np.expand_dims(testx_2, axis=1)

testx_3, testy_3 = read_csi(test_folder_name+'/m3')
testx_3 = np.expand_dims(testx_3, axis=1)
result_out_name = 'results/result_A1' + test_env +'_' + data_folder + '.pt'
#%%

print(data_folder + ': ' 'trained on '+ train_env + ', testing on ' + test_env)
n_way = 4
n_support = 5
n_query = 5

test_episode = 100

CF, acc = test(model1, model2, model3, testx_1, testy_1, testx_2, testy_2, testx_3, testy_3, n_way, n_support, n_query, test_episode)
#%%
result = {'CF':CF, 'acc':acc}
torch.save(result, result_out_name)
#%%
torch.save({
            'model1_state_dict': model1.state_dict(),
            'model2_state_dict': model2.state_dict(),
            'model3_state_dict': model3.state_dict(),
            'optimizer1_state_dict': optimizer1.state_dict(),
            'optimizer2_state_dict': optimizer2.state_dict(),
            'optimizer3_state_dict': optimizer3.state_dict(),
            }, model_out_name)
