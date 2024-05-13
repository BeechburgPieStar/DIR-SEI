import torch
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as scio
import random

def Power_Normalization(x):
    for i in range(x.shape[0]):
        max_power = np.sum((np.power(x[i,0,:],2) + np.power(x[i,1,:],2)))/x.shape[2]
        x[i] = x[i] / np.power(max_power, 1/2)
    return x

def TrainDataset():
    x = np.load(f'dataset/ADS-B/X_train_10Class.npy')
    x = Power_Normalization(x)
    x = np.squeeze(x)
    y = np.load(f'dataset/ADS-B/Y_train_10Class.npy')
    y = np.squeeze(y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3, random_state = 2023)

    return x_train, x_val, y_train, y_val


def TestDataset():
    x = np.load(f'dataset/ADS-B/X_test_10Class.npy')
    x = Power_Normalization(x)
    x = np.squeeze(x)
    y = np.load(f'dataset/ADS-B/Y_test_10Class.npy')
    y = np.squeeze(y)
    return x, y
