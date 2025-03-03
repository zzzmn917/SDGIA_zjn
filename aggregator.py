import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy



class Aggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout, act, name=None):
        super(Aggregator, self).__init__()
        self.dropout = dropout  
        self.act = act  
        self.batch_size = batch_size
        self.dim = dim  

    def forward(self):
        pass


class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0.,  step=1, name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim  
        self.dropout = dropout
        self.step = step
        
        self.w_ih = nn.Parameter(torch.Tensor(3 *self.dim, 2 *self.dim))
        self.w_hh = nn.Parameter(torch.Tensor(3 *self.dim, self.dim))
        self.b_ih = nn.Parameter(torch.Tensor(3 *self.dim))
        self.b_hh = nn.Parameter(torch.Tensor(3 *self.dim))
        self.b_oah = nn.Parameter(torch.Tensor(self.dim))
        self.b_iah = nn.Parameter(torch.Tensor(self.dim))

        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.linear_edge_in = nn.Linear(self.dim, self.dim, bias=True)
        self.linear_edge_out = nn.Linear(self.dim, self.dim, bias=True)
        self.linear_edge_f = nn.Linear(self.dim, self.dim, bias=True)

    def GNNCell(self, hidden, adj, mask_item=None):
        h = hidden  
        input_in = torch.matmul(adj[:, :, :adj.shape[1]], self.linear_edge_in(h)) + self.b_iah
        input_out = torch.matmul(adj[:, :, adj.shape[1]: 2 * adj.shape[1]], self.linear_edge_out(h)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(h, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)  
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (h - newgate)
        return hy

    def forward(self, hidden, adj, mask_item=None):
        for i in range(self.step):
            hidden = self.GNNCell( hidden, adj, mask_item=None)
        return hidden    


class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu, name=None):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act  
        self.dim = dim

        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim)) 
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))       
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))   


    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            alpha = torch.matmul(torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1)*neighbor_vector, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)  
            alpha = torch.softmax(alpha, -1).unsqueeze(-1)   
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)  
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        output = torch.cat([self_vectors, neighbor_vector], -1)
        # 对拼接后的表示进行 dropout 操作，以减少过拟合
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.w_3)
        output = output.view(batch_size, -1, self.dim)
        output = self.act(output)
        return output

