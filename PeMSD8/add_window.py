import numpy as np
import scipy.sparse as sp

def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def Add_Edge_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []  # windows
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index + window])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index + window])
            index = index + 1
    X = np.array(X)

    return X

def PI_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            index = index + 1
    X = np.array(X)
    return X

def Add_ZFC_Window_Horizon(data, data_type, window=3, horizon=1, alpha = 0.5, NVertices = None, scaleParameter = None, maxDimHoles = None, single=False,
                          train=True, val=False, val_ratio=None, test_ratio=None):
    length = len(data)
    end_index = length - horizon - window + 1
    X = []
    index = 0
    if single:
        while index < end_index:
            print(index)
            zigzag_PD = zigzag_persistence_diagrams(dataset = data_type, index= index, alpha = alpha,
                                                    NVertices = NVertices, scaleParameter = scaleParameter,
                                                    maxDimHoles = maxDimHoles, sizeWindow = window,
                                                    train=train, val=val, val_ratio= val_ratio, test_ratio = test_ratio)
            zigzag_fc = zigzag_filtration_curves(zigzag_PD)
            X.append(zigzag_fc)
            index = index + 1
    else:
        while index < end_index:
            print(index)
            zigzag_PD = zigzag_persistence_diagrams(dataset = data_type, index=index, alpha=alpha,
                                                    NVertices=NVertices, scaleParameter=scaleParameter,
                                                    maxDimHoles=maxDimHoles, sizeWindow=window,
                                                    train=train, val=val, val_ratio=val_ratio, test_ratio=test_ratio)
            zigzag_fc = zigzag_filtration_curves(zigzag_PD)
            X.append(zigzag_fc)
            index = index + 1

    X_array = np.array(X)
    return X_array
