

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score
import itertools


def bivariate_scatter_plot(ds, cols):
    l = len(cols)
    i = 0
    fig, ax = plt.subplots(l, (l)*2, sharex='row', sharey='row', figsize=((l)*6, (l)*3))
    for i in range(0, l):
        for j in range(0, l):
            if i == j:
                continue
            col1 = cols[i]
            col2 = cols[j]
            X = np.array(ds["X_train"][[col1, col2]])
            y = np.array(ds["y_train_b"])
            normal = X[np.argwhere(y==0)]
            attack = X[np.argwhere(y==1)]


            ax[j, i*2].scatter([s[0][0] for s in attack], [s[0][1] for s in attack], s = 25, color = 'red', alpha=.1)
            ax[j, i*2].set_xlabel(col1)
            ax[j, i*2].set_ylabel(col2)

            ax[j, (i*2)+1].scatter([s[0][0] for s in normal], [s[0][1] for s in normal], s = 25, color = 'cyan', alpha=.1)
            ax[j, (i*2)+1].set_xlabel(col1)
            ax[j, (i*2)+1].set_ylabel(col2)



    plt.show()


def histogram(ds, cols):
    X = np.array(ds["X_train"][cols])
    y = np.array(ds["y_train_b"])
    normal = X[np.argwhere(y == 0).flatten()]
    attack = X[np.argwhere(y == 1).flatten()]
    n, bins, patches = plt.hist(normal, 10, normed=1, facecolor='g', alpha=0.4)
    n, bins, patches = plt.hist(attack, 10, normed=1, facecolor='r', alpha=0.4)
    plt.xlabel(cols[0])
    plt.ylabel('Probability')
    plt.show()