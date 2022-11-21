import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
path = os.getcwd()


def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join(path + '/data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0:3]  #three dimensions, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join(path + '/data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0:3]  #three dimensions, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data


def load_topo_dataset(topo_data_type):
    # topo_data_type is for the data
    topo_train_data = np.load(path + '/data/PEMS08/PEMSD' + topo_data_type + '_ZFC_' + 'train' + '_data.npz', allow_pickle = True)['arr_0']
    topo_val_data = np.load(path + '/data/PEMS08/PEMSD' + topo_data_type + '_ZFC_' + 'validation' + '_data.npz', allow_pickle=True)['arr_0']
    topo_test_data = np.load(path + '/data/PEMS08/PEMSD' + topo_data_type + '_ZFC_' + 'test' + '_data.npz', allow_pickle=True)['arr_0']

    return topo_train_data, topo_val_data, topo_test_data


def load_edge_features(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join(path + '/data/PEMS04/PEMSD4_edge_features_matrix.npz')
        data = np.load(data_path, allow_pickle = True)['arr_0']
        hodge_laplacian = np.expand_dims(np.load(path + '/data/PEMS04/PEMSD4_hodge_Laplacian.npz', allow_pickle = True)['arr_0'], axis=0)
        incidence_matrix = np.expand_dims(np.load(path + '/data/PEMS04/PEMSD4_B1.npz',  allow_pickle = True)['arr_0'], axis=0)
    elif dataset == 'PEMSD8':
        data_path = os.path.join(path + '/data/PEMS08/PEMSD8_edge_features_matrix.npz')
        data = np.load(data_path, allow_pickle = True)['arr_0']
        hodge_laplacian = np.expand_dims(np.load(path + '/data/PEMS08/PEMSD8_hodge_Laplacian.npz')['arr_0'], axis=0)
        incidence_matrix = np.expand_dims(np.load(path + '/data/PEMS08/PEMSD8_B1.npz')['arr_0'], axis=0)
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Edge Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data, hodge_laplacian, incidence_matrix