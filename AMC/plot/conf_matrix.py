import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# Set the font family to Arial
plt.rcParams['font.family'] = 'Arial'


def plot_confusion_matrix(conf_mat, classes, normalize=True, title=None, cmap=plt.cm.Blues):
    title_fontsize = 22
    xlabel_fontsize = 20
    ylabel_fontsize = 20
    xticks_fontsize = 18
    yticks_fontsize = 18

    if not title:
        if normalize:
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix, without Normalization'

    cm = np.array(conf_mat)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Confusion Matrix, without Normalization")


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Show all ticks and label them with their respective list entries.
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',)

    # Set the title with the specified fontsize.
    ax.set_title(title, fontsize=22)
    ax.set_xlabel('Predicted label', fontsize=xlabel_fontsize)
    ax.set_ylabel('Predicted label', fontsize=ylabel_fontsize)
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
    plt.show()