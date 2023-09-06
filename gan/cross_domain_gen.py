'''
Cross domain data generator
e.g.) Using domain A generator Walk to Fall.
      Collect domain B Walk CSI data & Empty CSI data set.
      Generatre domain B Fall CSI data with domain B Walk noise + Empty CSI data.
'''
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from runner.utils import get_config
from models.discogan import Generator

model_path = './gan_checkpoint/model_gen_B-4.0'
data_path = './csi_dataset'

# Load trained generator
generator = torch.load(model_path)

# Load domain B Walk noise
walk_noise_B = np.load(os.path.join(data_path, 'noise/domain_B/walk_noise.npy'))

# Load domain B Empty CSI dataset
cfg = get_config('config.yaml')
win_size = cfg["window_size"]
batch_size = 200

empty_data = pd.read_csv(os.path.join(data_path, 'domain_B/empty.csv'))

total_num_wins = len(empty_data) // win_size
total_num_batch = len(walk_noise_B) // batch_size

generator.eval()

target_data = list()

print("Generate target data...")
for i in tqdm(range(total_num_wins)):
    empty_window = empty_data.iloc[i*win_size:i*win_size + win_size, 2:]
    empty_window = empty_window.astype(complex)
    empty_window = empty_window.apply(lambda x:x.abs())
    empty_window = np.array(empty_window)

    for j in range(2):
        target_noise_B = generator(torch.from_numpy(walk_noise_B[j*batch_size:j*batch_size + batch_size]).cuda().unsqueeze(1).float())
        for k in range(batch_size):
            target_data.append(empty_window + target_noise_B[k].squeeze().cpu().data.numpy())
    
print(f'Total data size: {len(target_data)}')
os.makedirs(os.path.join(data_path, 'fake_data'), exist_ok=True)
np.save(os.path.join(data_path, f'fake_data/fake_sit.npy'), np.array(target_data))

