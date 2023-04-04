"""
Author: Niloofar Bahadori
ReWiS for 1 monitor

Training:
    1) Read csi_data and, 2)Learning the embedding through ResNet 12 and 3) ProtoNet and Few-shot
Testing:
    1) Read csi_data and, 2)Utilizie the learned embedding function 12 and 3) ProtoNet for predition

Few-shot learning for classifying human activities for the case of one monitor.
This code require embedding training which is performed through the modified resnet code.
To train a model, load any amc_dataset start with m1c1_xxx or m1c4_xxx.
The code trains the model with the data collected from environment A1 and can be tested on either A2 or A3.
"""
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
from resnet import resnet12
#%%
def read_mat(csi_directory_path, csi_action):
    """
    Reads all the actions from a given activity directory
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
"""
 Load the amc_dataset's folder:either m3c1_xxx or m3c4_xxx ###
"""
data_folder = 'extractd_3x4/m1c4_PCA_80_300_extracted_3x4'
train_env = 'A1'
train_folder_name = 'few_shot_datasets/' + data_folder + '/train_A1'

model_out_name = 'models/model_' + data_folder.split('/')[-1] + '.pt'


trainx, trainy = read_csi(train_folder_name)
trainx = np.expand_dims(trainx, axis=1)
#%%
def extract_sample(n_way, n_support, n_query, datax, datay, test = False):
  """
  Picks random sample of size n_support+n_querry, for n_way classes
  Args:
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      datax (np.array): amc_dataset of dataloader dataframes
      datay (np.array): amc_dataset of labels
  Returns:
      (dict) of:
        (torch.Tensor): sample of dataloader dataframes. Size (n_way, n_support+n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
  """
  sample = []
  if test:
      K = np.array(['empty', 'jump', 'stand', 'walk'])
  else:
      K = np.random.choice(np.unique(datay), n_way, replace=False)
  for cls in K:
    datax_cls = datax[datay == cls]
    perm = np.random.permutation(datax_cls)
    sample_cls = perm[:(n_support+n_query)]
    sample.append(sample_cls)
  sample = np.array(sample)
  sample = torch.from_numpy(sample).float()
  return({
      'csi_mats': sample,
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query
      })


#sample_example = extract_sample(2, 8, 5, trainx, trainy)

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

def load_protonet_conv():
  """model = TheModelClass(*args, **kwargs)
  model.load_state_dict(torch.load(PATH))
  model.eval()
  Loads the prototypical network model
  Arg:
      x_dim (tuple): dimension of input dataloader dataframes
      hid_dim (int): dimension of hidden layers in conv blocks
      z_dim (int): dimension of embedded dataloader dataframes
  Returns:
      Model (Class ProtoNet)
  """
  ### Load the trained embedding model
  net = resnet12(avg_pool=True, num_classes = 4)
  net.load_state_dict(torch.load(model_out_name))

  
    
  return ProtoNet(net)


class ProtoNet(nn.Module):
  def __init__(self, net):
    """
    Args:
        encoder : CNN encoding the dataloader dataframes in sample
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
    """
    super(ProtoNet, self).__init__()
    self.net = net.cuda()

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
   
    z = self.net(x)
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
        'y_hat': y_hat
        # ,'target':target
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
      train_x (np.array): dataloader dataframes of training set
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
      sample = extract_sample(n_way, n_support, n_query, train_x, train_y, test = False)
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


model = load_protonet_conv()

optimizer = optim.Adam(model.parameters(), lr = 0.001)

n_way = 4
n_support = 5
n_query = 5

train_x = trainx
#train_x = torch.from_numpy(trainx).unsqueeze(0)
train_y = trainy

max_epoch = 2
epoch_size = 1000

train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size)
#%%

def test(model, test_x, test_y, n_way, n_support, n_query, test_episode):
  """
  Tests the protonet
  Args:
      model: trained model
      test_x (np.array): dataloader dataframes of testing set
      test_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      test_episode (int): number of episodes to test on
  """
  conf_mat = torch.zeros(n_way, n_way)
  running_loss = 0.0
  running_acc = 0.0
  for episode in tnrange(test_episode):
    sample = extract_sample(n_way, n_support, n_query, test_x, test_y, test = True)
    loss, output = model.set_forward_loss(sample)
    a = output['y_hat'].cpu().int()
    for cls in range(n_way):
        conf_mat[cls,:] = conf_mat[cls,:] + torch.bincount(a[cls,:], minlength=n_way) 
        
    running_loss += output['loss']
    running_acc += output['acc']
  avg_loss = running_loss / test_episode
  avg_acc = running_acc / test_episode
  print('Test results -- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))
  return (conf_mat/(n_query*test_episode), avg_acc)
  
# %%
"""
Pick the testing environment: either A2 or A3
"""
test_env = 'A2'
test_folder_name = 'few_shot_datasets/' + data_folder + '/test_' + test_env
testx, testy = read_csi(test_folder_name)
testx = np.expand_dims(testx, axis=1)

result_out_name = 'results/result_A1' + test_env +'_' + data_folder + '.pt'
  
n_way = 4
n_support = 5
n_query = 5

test_x = testx
test_y = testy

test_episode = 1
print(data_folder + ': ' 'trained on '+ train_env + ', testing on ' + test_env)
CF, acc = test(model, test_x, test_y, n_way, n_support, n_query, test_episode)
#%% saving model and results
result = {'CF':CF, 'acc':acc}
# torch.save(result, result_out_name)
# torch.save(model.state_dict(), model_out_name)
