import pandas as pd
import matplotlib.pyplot as plt
from plot_cfg import *


# Read csv
df1 = pd.read_csv('csv/testB_add1_15shot_result.csv')
df2 = pd.read_csv('csv/testB_add3_15shot_result.csv')
df3 = pd.read_csv('csv/testB_add5_15shot_result.csv')

snr_range = range(-20, 21, 2)
results1 = df1.mean(axis='columns').to_list()
results2 = df2.mean(axis='columns').to_list()
results3 = df3.mean(axis='columns').to_list()

plt.rcParams['font.family'] = 'Arial'

markers = ['*', '>', 'x']
colors = ['C0', 'C1', 'C2']
line_type = ['solid', 'dashed', 'dashdot']

plt.figure(figsize=(15, 12))


plt.plot(snr_range, results1, label=f'testB+1way', color=colors[0],
         marker=markers[0], linestyle=line_type[0], lw=lw, markersize=markersize, mew=mew)
plt.plot(snr_range, results2, label=f'testB+3way', color=colors[1],
         marker=markers[1], linestyle=line_type[1], lw=lw, markersize=markersize, mew=mew)
plt.plot(snr_range, results3, label=f'testB+5way', color=colors[2],
         marker=markers[2], linestyle=line_type[2], lw=lw, markersize=markersize, mew=mew)


plt.xlabel("Signal to Noise Ratio", fontsize=xlabel_fontsize)
plt.ylabel("Classification Accuracy", fontsize=ylabel_fontsize)
plt.xticks(fontsize=xticks_fontsize)
plt.yticks(fontsize=yticks_fontsize)
plt.legend(framealpha=1, fontsize=legend_fontsize)
plt.grid()

plt.savefig('figures/unseen_add_15shot.png', bbox_inches='tight')
