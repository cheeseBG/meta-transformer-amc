import pandas as pd
import matplotlib.pyplot as plt
from plot_cfg import *

# Read csv
columns = ['10to14', '10to18', '10to20', '10to30']
df = pd.read_csv('csv/snr_range_test.csv')

snr_range = range(-20, 21, 2)
results = [df[i].to_list() for i in columns]
print(results)


plt.rcParams['font.family'] = 'Arial'

markers = ['*', '>', 'x', '.']

for i, model in enumerate(columns):
    plt.plot(snr_range, results[i], label=f'{model}', marker=markers[i],
             markersize=16)

plt.xlabel("Signal to Noise Ratio", fontsize=xlabel_fontsize)
plt.ylabel("Classification Accuracy", fontsize=ylabel_fontsize)
plt.xticks(fontsize=xticks_fontsize)
plt.yticks(fontsize=yticks_fontsize)
plt.legend(loc='lower right', framealpha=1, fontsize=legend_fontsize)

plt.grid()
plt.show()