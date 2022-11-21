import torch
import torch.nn as nn
import torch.nn.functional as F
from ZFCSHGCN import ZFCSHGCNCNN

# GRU with the output from ZFCSHGCN
class ZFCSHGCNCNNCell(nn.Module):
    def __init__(self, node_num, edge_num, dim_in, dim_out, window_len, link_len, embed_dim):
        super(ZFCSHGCNCNNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = ZFCSHGCNCNN(dim_in+self.hidden_dim, 2*dim_out, window_len, link_len, embed_dim, node_num, edge_num)
        self.update = ZFCSHGCNCNN(dim_in+self.hidden_dim, dim_out, window_len, link_len, embed_dim, node_num, edge_num)

    def forward(self, x, state, x_full, node_embeddings, fixed_adj, adj, stay_cost, jump_cost, ZFC, hodge_Laplacian, x_e_window, incidence_matrix):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1) #x + state
        z_r = torch.sigmoid(self.gate(input_and_state, x_full, node_embeddings, fixed_adj, adj, stay_cost, jump_cost, ZFC, hodge_Laplacian, x_e_window, incidence_matrix))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, x_full, node_embeddings, fixed_adj, adj, stay_cost, jump_cost, ZFC, hodge_Laplacian, x_e_window, incidence_matrix))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
