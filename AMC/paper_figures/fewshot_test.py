import pandas as pd
import matplotlib.pyplot as plt
from plot_cfg import *


# Read csv
df1 = pd.read_csv('csv/testB_1shot_result.csv')
df2 = pd.read_csv('csv/testB_result.csv')
df3 = pd.read_csv('csv/testB_10shot_result.csv')
df4 = pd.read_csv('csv/testB_15shot_result.csv')

snr_range = range(-20, 21, 2)
results1 = df1.mean(axis='columns').to_list()
results2 = df2.mean(axis='columns').to_list()
results3 = df3.mean(axis='columns').to_list()
results4 = df4.mean(axis='columns').to_list()

plt.rcParams['font.family'] = 'Arial'

markers = ['*', '>', 'x', '.']
colors = ['C0', 'C1', 'C2', 'C3']
line_type = ['solid', 'dashed', 'dashdot', 'dotted']

plt.figure(figsize=(15, 12))


plt.plot(snr_range, results1, label=f'5way 1shot', color=colors[0],
         marker=markers[0], linestyle=line_type[0], lw=lw, markersize=markersize, mew=mew)
plt.plot(snr_range, results2, label=f'5way 5shot', color=colors[1],
         marker=markers[1], linestyle=line_type[1], lw=lw, markersize=markersize, mew=mew)
plt.plot(snr_range, results3, label=f'5way 10shot', color=colors[2],
         marker=markers[2], linestyle=line_type[2], lw=lw, markersize=markersize, mew=mew)
plt.plot(snr_range, results4, label=f'5way 15shot', color=colors[3],
         marker=markers[3], linestyle=line_type[3], lw=lw, markersize=markersize, mew=mew)


plt.xlabel("Signal to Noise Ratio", fontsize=xlabel_fontsize)
plt.ylabel("Classification Accuracy", fontsize=ylabel_fontsize)
plt.xticks(fontsize=xticks_fontsize)
plt.yticks(fontsize=yticks_fontsize)
plt.legend(framealpha=1, fontsize=legend_fontsize)
plt.grid()

plt.savefig('figures/fewshot.png', bbox_inches='tight')
