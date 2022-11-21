import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.spatial.distance import pdist,squareform
from GNN import GNN
import numpy as np


class ZFCSHGCNCNN(nn.Module):
    def __init__(self, dim_in, dim_out, window_len, link_len, embed_dim, node_num, edge_num):
        super(ZFCSHGCNCNN, self).__init__()
        self.link_len = link_len
        self.edge_num = edge_num
        # for spatial graph convolution
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, link_len, dim_in, int(dim_out/4)))
        # for supra-Laplacian graph convolution
        if (dim_in - 1) % 16 == 0:  # (dim_in-3)%16 ==0
            self.window_weights_supra = nn.Parameter(torch.FloatTensor(embed_dim, 1, int(dim_out / 8)))
        else:  # dim_in == dim_out
            self.window_weights_supra = nn.Parameter(torch.FloatTensor(embed_dim, int(dim_in / 2), int(dim_out / 8)))
        # for temporal graph convolution
        if (dim_in - 1) % 16 == 0:  # (dim_in-3)%16 ==0
            self.window_weights_temporal = nn.Parameter(torch.FloatTensor(embed_dim, 1, int(dim_out / 2))) # /2
        else:  # dim_in == dim_out
            self.window_weights_temporal = nn.Parameter(torch.FloatTensor(embed_dim, int(dim_in / 2), int(dim_out / 2))) # /2
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.T = nn.Parameter(torch.FloatTensor(window_len)) #did not use it in the model
        self.gnn = GNN(int(dim_out / 8), int(node_num))
        self.lin = torch.nn.Linear(int(1), int(1)) # where dim_e_in = 1

    def forward(self, x, x_window, node_embeddings, fixed_adj, adj, stay_cost, jump_cost, ZFC_input, hodge_Laplacian, x_e_window, incidence_matrix):
        # incidence matrix is B1
        node_num = node_embeddings.shape[0]
        initial_S = F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))) # (N, N); ! # add .clone() for new version
        initial_S.fill_diagonal_(0)
        initial_S.fill_diagonal_(stay_cost) # probability to stay at the same layer
        if fixed_adj is True:
            S = torch.FloatTensor(np.array([adj[i, j] * np.exp(initial_S[i, j])
                                            for i in range(node_num) for j in range(node_num)]).reshape(node_num, node_num))
        else:
            S = F.softmax(initial_S, dim = 1)
        S = (S/torch.sum(S, dim = 1)).to(node_embeddings.device) # (N, N) which can be directly used as Laplacian
        support_set = [torch.eye(node_num).to(S.device), S]

        for k in range(2, self.link_len):
            support_set.append(torch.mm(S, support_set[k-1]))
        supports = torch.stack(support_set, dim=0)

        T = x_window.size(1)
        Bootstrap_num = np.random.choice(range(T), size=(3,))  # randomly select three elements in the window
        Bootstrap_num.sort()
        supra_laplacian = torch.zeros(size=(self.edge_num * Bootstrap_num.shape[0], self.edge_num * Bootstrap_num.shape[0])).to(S.device)
        inter_diagonal_matrix = np.zeros(shape=(self.edge_num, self.edge_num), dtype=np.float32)
        np.fill_diagonal(inter_diagonal_matrix, jump_cost)
        inter_diagonal_matrix = torch.FloatTensor(inter_diagonal_matrix).to(S.device)
        # layer 0 -> layer 1, ..., layer 0 -> layer L; layer 1 -> layer 2, ..., layer 2 -> layer L
        for i in range(Bootstrap_num.shape[0]):
            for j in range(Bootstrap_num.shape[0]):
                if i == j:
                    supra_laplacian[self.edge_num * i: self.edge_num * (i + 1), self.edge_num * i: self.edge_num * (i + 1)] = hodge_Laplacian
                elif j > i:
                    supra_laplacian[self.edge_num * i: self.edge_num * (i + 1),
                    self.edge_num * j: self.edge_num * (j + 1)] = inter_diagonal_matrix

        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool) #(N, link_len, dim_in, dim_out/4)
        bias = torch.matmul(node_embeddings, self.bias_pool) #(N, dim_out)
        x_s = torch.einsum("knm,bmc->bknc", supports, x) #(B, link_len, N, dim_in)
        x_s = x_s.permute(0, 2, 1, 3) #(B, N, link_len, dim_in)
        x_sconv = torch.einsum('bnki,nkio->bno', x_s, weights) #(B, N, dim_out/4)

        weights_window_supra = torch.einsum('nd,dio->nio', node_embeddings, self.window_weights_supra)  # (N, dim_in, dim_out/4)
        x_e_window_ = x_e_window[:, Bootstrap_num, :, :]
        _x_e_window_ = x_e_window_.view(x_e_window_.size(0), -1, x_e_window_.size(3))  # (B, num_edges*Bootstrap_num, dim_e_in)
        _x_e_window_ = self.lin(_x_e_window_)  # (B, num_edges*Bootstrap_num, dim_e_in)
        x_e_w_s = torch.einsum('bmi,mn->bni', _x_e_window_, supra_laplacian)  # (B, num_edges*Bootstrap_num, dim_e_in)
        x_e_w_s = x_e_w_s.view(x_e_w_s.size(0), Bootstrap_num.shape[0], -1, self.edge_num) # (B, Bootstrap_num, dim_e_in, num_edges)
        # edge to node aggregation
        x_e_n_w_s = torch.einsum('bfim,mn->bfin', x_e_w_s, incidence_matrix.transpose(0,1)) # (B, Bootstrap_num, dim_e_in, num_nodes)
        x_e_n_w_s = x_e_n_w_s.view(x_e_n_w_s.size(0), Bootstrap_num.shape[0], node_num, -1) # (B, Bootstrap_num, num_nodes, dim_e_in)
        x_e_n_wconv_s = torch.einsum('bfni,nio->bfno', x_e_n_w_s, weights_window_supra)  # (B, Bootstrap_num, num_edges, dim_out/4)
        # global mean pooling
        x_e_n_wconv_s = torch.mean(x_e_n_wconv_s, dim=1)  # (B, N, dim_out/4)

        weights_window_temporal = torch.einsum('nd,dio->nio', node_embeddings, self.window_weights_temporal)  #(N, dim_in, dim_out/4)
        x_w_t = torch.einsum('btni,nio->btno', x_window, weights_window_temporal)  #(B, T, N, dim_out/4)
        x_w_t = x_w_t.permute(0, 2, 3, 1)  #(B, N, dim_out/4, T)
        x_wconv_t = torch.matmul(x_w_t, self.T)  #(B, N, dim_out/4)

        ZFC_input = ZFC_input.view(ZFC_input.size(0), -1, 1) # (B, 50) -> (B, 50, 1), where 50 = 25 * 2
        ZFC = ZFC_input.repeat(1,1, node_num) # (B, 50, node_num)
        ZFC_output = self.gnn(ZFC) # (B, num_state, node_num)
        x_ZFC = ZFC_output.view(ZFC.size(0), node_num, -1) # #(B, N, dim_out/4)

        x_tswconv = torch.cat([x_sconv, x_ZFC, x_wconv_t, x_e_n_wconv_s], dim = -1) + bias #(B, N, dim_out)
        return x_tswconv
