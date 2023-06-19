import pandas as pd
import matplotlib.pyplot as plt
from plot_cfg import *


# Read csv
columns = ['ResNet', 'CNN', 'ProtoNet', 'Proposed']
df1 = pd.read_csv('csv/learning_-20to20.csv')
df2 = pd.read_csv('csv/learning_-10to20.csv')
df3 = pd.read_csv('csv/learning_0to10.csv')
df4 = pd.read_csv('csv/learning_0to20.csv')

snr_range = range(-20, 21, 2)
results1 = [df1[i].to_list() for i in columns]
results2 = [df2[i].to_list() for i in columns]
results3 = [df3[i].to_list() for i in columns]
results4 = [df4[i].to_list() for i in columns]

plt.rcParams['font.family'] = 'Arial'

markers = ['*', '>', 'x', '.']
colors = ['C0', 'C1', 'C2', 'C4']
line_type = ['solid', 'dashed', 'dashdot', 'dotted']

plt.figure(figsize=(15, 12))

for i, model in enumerate(columns):
    plt.plot(snr_range, results1[i], label=f'{model} -20to20', color=colors[i],
             marker=markers[0], linestyle=line_type[0], lw=lw, markersize=markersize, mew=mew)
    plt.plot(snr_range, results2[i], label=f'{model} -10to20', color=colors[i],
             marker=markers[1], linestyle=line_type[1], lw=lw, markersize=markersize, mew=mew)
    plt.plot(snr_range, results3[i], label=f'{model} 0to10', color=colors[i],
             marker=markers[2], linestyle=line_type[2], lw=lw, markersize=markersize, mew=mew)
    plt.plot(snr_range, results4[i], label=f'{model} 0to20', color=colors[i],
             marker=markers[3], linestyle=line_type[3], lw=lw, markersize=markersize, mew=mew)

plt.xlabel("Signal to Noise Ratio", fontsize=xlabel_fontsize)
plt.ylabel("Classification Accuracy", fontsize=ylabel_fontsize)
plt.xticks(fontsize=xticks_fontsize)
plt.yticks(fontsize=yticks_fontsize)
plt.legend(framealpha=1, fontsize=legend_fontsize, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.15))
plt.grid()

plt.savefig('figures/learning.png', bbox_inches='tight')
