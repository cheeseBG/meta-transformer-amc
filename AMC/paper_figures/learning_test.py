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