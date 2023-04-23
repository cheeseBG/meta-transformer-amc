import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
from data.dataset import AMCTrainDataset
from runner.utils import get_config

config = get_config('config.yaml')
train_data = AMCTrainDataset(config["dataset_path"], robust=False, mode='easy', snr_range=config["snr_range"])
data = [complex(val[0], val[1]) for val in zip(train_data.__getitem__(4024)['data'][0], train_data.__getitem__(4024)['data'][1])]

# STFT 설정 변수
window_size = 4  # 윈도우 크기
overlap = 0.5      # 50% 오버랩
n_fft = 256        # FFT 포인트 수

# STFT 계산을 위한 윈도우 이동 단계 계산
step_size = int(window_size * (1 - overlap))

# STFT 변환
freqs, times, Zxx = stft(data, fs=1.0, window='hann', nperseg=window_size, noverlap=step_size, nfft=n_fft)

# STFT 결과를 절댓값으로 변환 (복소수를 실수로 변환)
Zxx_abs = np.abs(Zxx)
print(Zxx_abs.shape)

# 스펙트로그램 시각화
plt.figure()
plt.pcolormesh(times, freqs, Zxx_abs, shading='gouraud')
plt.title("Spectrogram")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
plt.colorbar(label="Amplitude")
plt.show()
