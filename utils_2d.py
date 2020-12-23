#!/home/the42nd/anaconda3/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
import torch

sns.set()

def get_binary_tag(tag):
    if tag < split:
        return 0
    else:
        return 1

def get_binary_blobs(size = 100000, centers = 6, seed = 42):
    X, y = make_blobs(size, centers = centers, random_state = seed)
    df = pd.DataFrame(X, columns = ['x1', 'x2'])
    df['tag'] = y
    
    if type(centers) is list:
        num_centers = len(centers)
    else:
        num_centers = centers
    
    for i in range(num_centers):
        plt.scatter(df.loc[df.tag == i,'x1'].values, df.loc[df.tag == i,'x2'].values, label = i)
    plt.title("Blob Data")
    plt.legend()
    plt.show()
    
    global split
    split = int(num_centers / 2)

    df['binary_tag'] = df.tag.map(get_binary_tag)
    
    for i in range(2):
        plt.scatter(df.loc[df.binary_tag == i,'x1'].values, df.loc[df.binary_tag == i,'x2'].values, label = i)
    plt.title("Blobs split into 2 groups")
    plt.legend()
    plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(df[['x1', 'x2']].values, df.binary_tag.values, 
                                                     test_size=0.25, random_state = seed)
    
    for i in range(2):
        plt.scatter(X_train[y_train == i,0], X_train[y_train == i,1], label = i)
    plt.title("Train Set")
    plt.legend()
    plt.show()

    for i in range(2):
        plt.scatter(X_test[y_test == i,0], X_test[y_test == i,1], label = i)
    plt.title("Test Set")
    plt.legend()
    plt.show()
    
    return X_train, X_test, y_train, y_test

def plot_decision_boundary(model, X, y):
    if torch.cuda.is_available():
        pred = model(torch.Tensor(X).cuda()).cpu().detach().numpy()
    else:
        pred = model(torch.Tensor(X)).detach().numpy()
    auc = roc_auc_score(y, pred)
    print(f"AUC: {auc}")
    mse = mean_squared_error(y, pred)
    print(f"MSE: {mse}")
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    if torch.cuda.is_available():
        Z = model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).cuda())
        Z = Z.cpu().detach().numpy().reshape(xx.shape)
    else:
        Z = model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.detach().numpy().reshape(xx.shape)
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    Z = np.round(Z)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap = plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Spectral)
    plt.show()
    
    return auc, mse
