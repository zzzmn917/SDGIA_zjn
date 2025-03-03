import numpy as np
import torch
from torch.utils.data import Dataset



def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set  
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)  
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]  
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]  
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]  
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]  

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y) 


def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]  
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len  
    
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len



def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)  
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64) 
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)  # 无放回抽样
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)   # 有放回抽样
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity 


class Data(Dataset):
    def __init__(self, data, train_len=None):
        inputs, mask, max_len = handle_data(data[0], train_len)
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len


    def __getitem__(self, index):
        u_input, mask, targets = self.inputs[index], self.mask[index], self.targets[index] 
        max_n_node = self.max_len
        node = np.unique(u_input)
        items = node.tolist() + (max_n_node - len(node)) * [0] 
        adj1 = np.zeros((max_n_node, max_n_node))  
        for i in np.arange(len(u_input) - 1):
            # 邻接矩阵的对角线元素设置为 1
            u = np.where(node == u_input[i])[0][0]
            # adj1[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            # adj1[v][v] = 1
            adj1[u][v] = 1
        adj_sum_in = np.sum(adj1, 0)     
        adj_sum_in[np.where(adj_sum_in == 0)] = 1  
        adj_in = np.divide(adj1, adj_sum_in) 
        np.fill_diagonal(adj_in, 1)
        adj_sum_out = np.sum(adj1, 1)   
        adj_sum_out[np.where(adj_sum_out == 0)] = 1  
        adj_out = np.divide(adj1.transpose(), adj_sum_out)  
        np.fill_diagonal(adj_out, 1)
        adj1 = np.concatenate([adj_in, adj_out]).transpose()
        # self.adj.append(adj1)  # adj：归一化后的出入度邻接矩阵  
        
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]

        return [torch.tensor(alias_inputs), torch.tensor(adj1), torch.tensor(items),
                torch.tensor(mask), torch.tensor(targets), torch.tensor(u_input)]

    def __len__(self):
        return self.length
