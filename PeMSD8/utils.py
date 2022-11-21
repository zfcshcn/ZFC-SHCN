import numpy as np
import networkx as nx
import pandas as pd
from numpy.linalg import inv, pinv
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
import os
from incidence_matrix import get_faces, incidence_matrices

path = os.getcwd()

def get_adjacency_matrxix(dataset, number_nodes):
    PEMS_net_dataset = pd.read_csv(path + '/data/PEMS0' + str(dataset)[5] + '/distance.csv', header=0)
    PEMS_net_edges = PEMS_net_dataset.values[:, 0:2]
    A = np.zeros((number_nodes, number_nodes), dtype= np.float32)
    for i in range(PEMS_net_edges.shape[0]):
        A[int(PEMS_net_edges[i,0] -1 ), int(PEMS_net_edges[i,1] -1 )] = 1.
        A[int(PEMS_net_edges[i, 1] - 1), int(PEMS_net_edges[i, 0] -1 )] = 1.
    A = sp.csr_matrix(A)
    return A

# Fractional power
def fractional_fltr(adj, number_nodes, sigma, gamma):
    degrees = np.array(adj.sum(1)).flatten()
    degrees[np.isinf(degrees)] = 0.
    D = sp.diags(degrees, 0)
    L_darray = (D - adj).toarray()
    D, V = np.linalg.eigh(L_darray, 'U')
    M_gamma_Lambda = D
    M_gamma_Lambda[M_gamma_Lambda < 1e-5] = 0
    M_V = V
    M_gamma_Lambda = np.float_power(M_gamma_Lambda, gamma)
    M_gamma_Lambda = np.diag(M_gamma_Lambda, 0)
    M_gamma_Lambda = sp.csr_matrix(M_gamma_Lambda)
    M_V = sp.csr_matrix(M_V)
    Lg = M_V * M_gamma_Lambda
    Lg = Lg * sp.csr_matrix.transpose(M_V)
    Lg = Lg.toarray()
    Lg = Lg.reshape(1, -1)
    Lg[abs(Lg) < 1e-5] = 0.
    Lg = Lg.reshape(number_nodes, -1)
    Dg = np.diag(np.diag(Lg))
    Ag = Dg - Lg
    Ag = sp.csr_matrix(Ag)
    power_Dg_l = np.float_power(np.diag(Dg), -sigma)
    power_Dg_l = sp.csr_matrix(np.diag(power_Dg_l))
    power_Dg_r = np.float_power(np.diag(Dg), (sigma - 1))
    power_Dg_r = sp.csr_matrix(np.diag(power_Dg_r))
    fractional_fltr = power_Dg_l * Ag
    fractional_fltr = fractional_fltr * power_Dg_r
    return fractional_fltr

def node_corr_cosine(adj, feat):
    """calculate edge cosine distance"""
    # prod = np.dot(feat, feat.T)
    distance = squareform(pdist(feat, 'cosine'))
    edge_feat = distance[np.nonzero(sp.triu(adj, k=1))]
    ret = edge_feat.reshape((len(edge_feat), 1))
    return ret

def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)


def compute_D2(B):
    """
    Computes D2 = max(diag(dot(|B|, 1)), I)
    """
    B_rowsum = np.abs(B).sum(axis=1)

    D2 = np.diag(np.maximum(B_rowsum, 1))
    return D2

def compute_D1(B1, D2):
    """
    Computes D1 = 2 * max(diag(|B1|) .* D2
    """
    rowsum = (np.abs(B1) @ D2).sum(axis=1)
    D1 = 2 * np.diag(rowsum)

    return D1

def compute_bunch_matrices(B1, B2):
    """
    Computes normalized A0 and A1 matrices (up and down),
        and returns all matrices needed for Bunch model shift operators
    """
    # D matrices
    D2_2 = compute_D2(B2)
    D2_1 = compute_D2(B1)
    D3_n = np.identity(B1.shape[1]) # (|E| x |E|)
    D1 = compute_D1(B1, D2_2)
    D3 = np.identity(B2.shape[1]) / 3 # (|F| x |F|)

    # L matrices
    D1_pinv = pinv(D1)
    D2_2_inv = inv(D2_2)

    L1u = D2_2 @ B1.T @ D1_pinv @ B1
    L1d = B2 @ D3 @ B2.T @ D2_2_inv
    L1f = L1u + L1d

    return L1f


def compute_hodge_matrix(data):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(data.nodes))])
    edge_index_ = np.array(data.edges).transpose()
    edge_index = [(edge_index_[0, i], edge_index_[1, i]) for i in
                  range(np.shape(edge_index_)[1])]
    g.add_edges_from(edge_index)

    edge_to_idx = {edge: i for i, edge in enumerate(g.edges)}

    B1, B2 = incidence_matrices(g, sorted(g.nodes), sorted(g.edges), get_faces(g), edge_to_idx)

    return B1, B2


'''
# generate Hodge Laplacian for PEMSD4
PEMS_net_dataset = pd.read_csv(path + '/data/PEMS08' + '/distance.csv', header=0)
PEMS_net_edges = PEMS_net_dataset.values[:, 0:2]
PEMS_net_edgelist = [(int(u), int(v)) for u, v in PEMS_net_edges]
PEMS_net = nx.Graph()
PEMS_net.add_edges_from(PEMS_net_edgelist)

PEMSD8_B1, PEMSD8_B2 = compute_hodge_matrix(PEMS_net) # (170, 274), (274, 80)
print(PEMSD8_B1.shape)
print(PEMSD8_B2.shape)
PEMSD8_hodge_Laplacian = compute_bunch_matrices(PEMSD8_B1, PEMSD8_B2)
np.savez_compressed('PEMSD8_hodge_Laplacian.npz', PEMSD8_hodge_Laplacian)
np.savez_compressed('PEMSD8_B1.npz', PEMSD8_B1)
'''

'''
# edge feature
PEMS_net_dataset = pd.read_csv(path + '/data/PEMS08' + '/distance.csv', header=0)
PEMS_net_edges = PEMS_net_dataset.values[:, 0:2]
PEMS_net_edgelist = [(int(u), int(v)) for u, v in PEMS_net_edges]
PEMS_net = nx.Graph()
PEMS_net.add_edges_from(PEMS_net_edgelist)
PEMS_net_adj = nx.adjacency_matrix(PEMS_net).toarray()

data_path = os.path.join(path + '/data/PEMS08/PEMS08.npz')
data = np.load(data_path)['data'][:, :, 0:3]
edge_features_matrix = np.zeros(shape=(data.shape[0], len(PEMS_net.edges), 1)) # (num_obs, num_edges, 1): (16992, 340, 1)

for i in range(data.shape[0]):
    edge_features_matrix[i] = node_corr_cosine(PEMS_net_adj, data[i,:,:])

np.savez_compressed('PEMSD8_edge_features_matrix.npz', edge_features_matrix)
'''

'''
x = np.load('data/PEMS08/PEMSD8_ZFC_train_data.npz', allow_pickle=True)['arr_0']
zeros = np.zeros((2,2))
PEMSD8_ZFC_train_data = np.zeros((x.shape[0], 25, 2))
for i in range(x.shape[0]):
    if x[i].shape[0]!=25:
        PEMSD8_ZFC_train_data[i] = np.concatenate([x[i], zeros], axis=0)
    else:
        PEMSD8_ZFC_train_data[i] = x[i]

np.savez_compressed('PEMSD8_ZFC_complete_train_data.npz', PEMSD8_ZFC_train_data)
'''

#x = np.load('data/PEMS08/PEMSD8_edge_features_matrix.npz', allow_pickle=True)['arr_0'] (17856, 274, 1)
