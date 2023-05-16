import pandas as pd
import matplotlib.pyplot as plt
from plot_cfg import *


# Read csv
columns = [16, 32, 64, 128, 256, 512, 1024]
model_name = 'cnn'  # vit, cnn

df = pd.read_csv(f'csv/size_{model_name}.csv')

snr_range = range(-20, 21, 2)
results = [df[str(i)].to_list() for i in columns]

plt.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(12, 7))
markers = ['*', '>', 'x', '.', '^', '<', 'v']

for i, model in enumerate(columns):
    plt.plot(snr_range, results[i], label=f'sample_size {model}', marker=markers[i],
             markersize=markersize, mew=mew)

plt.xlabel("Signal to Noise Ratio", fontsize=xlabel_fontsize)
plt.ylabel("Classification Accuracy", fontsize=ylabel_fontsize)
plt.xticks(fontsize=xticks_fontsize)
plt.yticks(fontsize=yticks_fontsize)
plt.legend(loc='upper left', framealpha=1, fontsize=legend_fontsize)

plt.grid()

plt.savefig(f'figures/size_{model_name}.png', bbox_inches='tight')
