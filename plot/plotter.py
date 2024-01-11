import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# Set the font family to Arial
plt.rcParams['font.family'] = 'Arial'


def plot_confusion_matrix(conf_mat, classes, normalize=True, title=None, cmap=plt.cm.Blues):
    xlabel_fontsize = 32
    ylabel_fontsize = 32
    xticks_fontsize = 26
    yticks_fontsize = 26

    cm = np.array(conf_mat)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Confusion Matrix, without Normalization")

    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(im, ax=ax, cax=cax)

    # Show all ticks and label them with their respective list entries.
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',)

    # Set the title with the specified fontsize.
    ax.set_xlabel('Predicted label', fontsize=xlabel_fontsize)
    ax.set_ylabel('True label', fontsize=ylabel_fontsize)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=xticks_fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=yticks_fontsize)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig('paper_figures/figures/best_conf_resnet.png', bbox_inches='tight')

def eval_plotter(snr_range, acc_per_size, sample_size_list):
    # SNR Graph
    plt.rcParams['font.family'] = 'Arial'
    title_fontsize = 32
    xlabel_fontsize = 30
    ylabel_fontsize = 30
    xticks_fontsize = 28
    yticks_fontsize = 28
    legend_fontsize = 20

    markers = ['*', '>', 'x', '.', '^', '<', 'v']

    for i, sample_size in enumerate(sample_size_list):
        plt.plot(snr_range, acc_per_size[i], label=f'sample_size{str(sample_size)}', marker=markers[i],
                    markersize=16)

    plt.xlabel("Signal to Noise Ratio", fontsize=xlabel_fontsize)
    plt.ylabel("Classification Accuracy", fontsize=ylabel_fontsize)
    plt.title("Classification Accuracy on RadioML 2018.01 Alpha", fontsize=title_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.legend(loc='lower right', framealpha=1, fontsize=legend_fontsize)
    plt.show()