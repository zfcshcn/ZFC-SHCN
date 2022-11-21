import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from add_window import Add_Window_Horizon, Add_Edge_Window_Horizon
from load_dataset import load_st_dataset, load_topo_dataset, load_edge_features
from normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def triple_data_loader(X, ZFC, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, ZFC, Y = TensorFloat(X), TensorFloat(ZFC), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, ZFC, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def multi_data_loader(X, X_e, hodge_Laplacian, incidence_matrix, ZFC, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, X_e, hodge_Laplacian, incidence_matrix, ZFC, Y = TensorFloat(X), TensorFloat(X_e), TensorFloat(hodge_Laplacian), TensorFloat(incidence_matrix), TensorFloat(ZFC), TensorFloat(Y)
    data_0 = torch.utils.data.TensorDataset(X, X_e, ZFC, Y)
    dataloader_0 = torch.utils.data.DataLoader(data_0, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    data_1 = torch.utils.data.TensorDataset(hodge_Laplacian, incidence_matrix)
    dataloader_1 = torch.utils.data.DataLoader(data_1, batch_size=1, # 1 rather than batch_size
                                               shuffle=shuffle, drop_last=drop_last)
    return dataloader_0, dataloader_1

def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=False):
    #load raw st dataset
    data = load_st_dataset(args.dataset)        # B, N, D
    #normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)

    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)

    #add time window
    single = False
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single) # lag (i.e., window) = 12, horizon = 12, single = True
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler


def get_topo_dataloader(args, topo_data_type, normalizer = 'std', tod=False, dow=False, weather=False, single=False):
    #load raw st dataset
    data = load_st_dataset(args.dataset) #B, N, D
    #normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)

    # after window_horizon process for topological features
    topo_tra,topo_val,topo_test = load_topo_dataset(topo_data_type)

    # edge features
    edge_data, hodge_laplacian, incidence_matrix = load_edge_features(args.dataset)

    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
        edge_data_train, edge_data_val, edge_data_test = split_data_by_days(edge_data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
        edge_data_train, edge_data_val, edge_data_test = split_data_by_ratio(edge_data, args.val_ratio, args.test_ratio)

    single = False
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    edge_x_tra = Add_Edge_Window_Horizon(edge_data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    edge_x_val = Add_Edge_Window_Horizon(edge_data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    edge_x_test = Add_Edge_Window_Horizon(edge_data_test, args.lag, args.horizon, single)
    print('Train: x, x_e, ZFC, y ->', x_tra.shape, edge_x_tra.shape, topo_tra.shape, y_tra.shape)
    print('Val: x, x_e, ZFC, y ->', x_val.shape, edge_x_val.shape, topo_val.shape, y_val.shape)
    print('Test: : x, x_e, ZFC, y ->', x_test.shape, edge_x_test.shape, topo_test.shape, y_test.shape)
    ##############get triple dataloader######################
    train_dataloader_0, train_dataloader_1 = multi_data_loader(x_tra, edge_x_tra, hodge_laplacian, incidence_matrix, topo_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader_0 = None
        val_dataloader_1 = None
    else:
        val_dataloader_0, val_dataloader_1 = multi_data_loader(x_val, edge_x_val, hodge_laplacian, incidence_matrix, topo_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader_0, test_dataloader_1 = multi_data_loader(x_test, edge_x_test, hodge_laplacian, incidence_matrix, topo_test, y_test, args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader_0, train_dataloader_1, val_dataloader_0, val_dataloader_1, test_dataloader_0, test_dataloader_1, scaler