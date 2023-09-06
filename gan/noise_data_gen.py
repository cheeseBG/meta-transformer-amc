'''
%% noise data generator %%
- empty CSI data를 기준으로 나머지 class에 대한 noise 추출
- 가져오는 데이터 범위를 정한후, 일정 window size로 crop후 노이즈 추출
'''
import os
import numpy as np
import pandas as pd
from runner.utils import get_config

# Config
win_size = 64 # 시각적으로 확인하기 위해 64으로 설정 -> 이후에는 실험에 맞게 -> discriminator 수정 필요
config = get_config('config.yaml')
labels = config['activity_labels']
data_path = config['train_dataset_path']

noise_data_save_path = './csi_dataset/noise'


def generate_windows(df_dict, win_size=100, amp=True):
    global min_num_wd

    win_dict = dict()
    for atv in df_dict.keys():
        windows = list()
        df = df_dict[atv]
        num_win = len(df) // win_size

        for i in range(num_win):
            wd = df.iloc[i*win_size:i*win_size + win_size, 2:]
            wd = wd.astype(complex)
        
            if amp is True:
                wd = wd.apply(lambda x: x.abs())
            
            wd = wd.to_numpy()
            windows.append(wd)
        
        win_dict[atv] = np.array(windows)
    return win_dict


if __name__=="__main__":
    # Read CSI file and convert to dataframe
    data_df = dict()

    for atv in labels:
        f_path = os.path.join(data_path, atv + '.csv')
        data_df[atv] = pd.read_csv(f_path)

    # Shape of each class [num_window, window_size, num_subcarrier]
    window_data= generate_windows(data_df, win_size=win_size)

    # Save data (.npy format)
    base_data = window_data['empty']

    domain = data_path.split('/')[-1]
    save_path = os.path.join(noise_data_save_path, domain)
    os.makedirs(save_path, exist_ok=True)

    print("Generate noise data...")
    for label in window_data.keys():
        if label != 'empty':
            noise_list = list()
            for i in range(len(base_data)):
                extract_data = window_data[label]
                for j in range(len(extract_data)):
                    noise = extract_data[j] - base_data[i]
                    noise_list.append(noise)
            noise_list = np.array(noise_list)

            np.save(os.path.join(save_path, f'{label}_noise.npy'), noise_list)
            print(f'{label}_noise.npy is saved!')



