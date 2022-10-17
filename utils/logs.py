import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def plot_pic(array, epochs, title, x_label, y_label, doc_name):
    x = np.arange(1, epochs + 1)
    plt.xlabel(x_label)
    y = array
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(x, y)
    path = os.path.join('Log', doc_name)

    if os.path.exists(path) == False:
        os.makedirs(path)

    path = os.path.join(path, title+'.jpg')
    plt.savefig(path)

def add_log(log, key, value):
    if key not in log.keys():
        log[key] = []
    log[key].append(value)

def save_log(doc_name, log):
    path = os.path.join('Log', doc_name)
    if os.path.exists(path) == False:
        os.makedirs(path)
    with open(os.path.join(path, 'log.pkl'), 'wb') as f:
        pickle.dump(log, f)
