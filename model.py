import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator
# from aggregator import LocalAggregator, GlobalAggregator, TargetAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F



class FindNeighbors(Module):
    def __init__(self, dim):
        super(FindNeighbors, self).__init__()
        self.hidden_size = dim
        self.neighbor_n = 9   # Diginetica:3; Tmall: 9; Nowplaying: 3
        self.dropout40 = nn.Dropout(0.60)   # Diginetica:0.80;  Tmall:0.60  Nowplaying:0.80

    def compute_sim(self, sess_emb):
        fenzi = torch.matmul(sess_emb, sess_emb.permute(1, 0)) 
        fenmu_l = torch.sum(sess_emb * sess_emb + 0.000001, 1)
        fenmu_l = torch.sqrt(fenmu_l).unsqueeze(1)
        fenmu = torch.matmul(fenmu_l, fenmu_l.permute(1, 0))
        cos_sim = fenzi / fenmu 
        cos_sim = nn.Softmax(dim=-1)(cos_sim)
        return cos_sim

    def forward(self, sess_emb):
        k_v = self.neighbor_n 
        cos_sim = self.compute_sim(sess_emb) 
        if cos_sim.size()[0] < k_v:
            k_v = cos_sim.size()[0]
        cos_topk, topk_indice = torch.topk(cos_sim, k=k_v, dim=1)
        cos_topk = nn.Softmax(dim=-1)(cos_topk)
        sess_topk = sess_emb[topk_indice]

        cos_sim = cos_topk.unsqueeze(2).expand(cos_topk.size()[0], cos_topk.size()[1], self.hidden_size)

        neighbor_sess = torch.sum(cos_sim * sess_topk, 1)
        neighbor_sess = self.dropout40(neighbor_sess)  # [b,d]
        return neighbor_sess



class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt
        self.beta = opt.beta
        self.batch_size = opt.batch_size
        self.num_node = num_node 
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local    
        self.dropout_global = opt.dropout_global  
        self.hop = opt.n_iter  
        self.sample_num = opt.n_sample  
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()   
        self.num = trans_to_cuda(torch.Tensor(num)).float() 
        
        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0, step=opt.step)
        # self.target_agg = TargetAggregator(self.dim, self.opt.alpha, opt.dropout_target)
        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # 项目嵌入 & 位置嵌入
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # target
        self.FindNeighbor = FindNeighbors(self.dim)
        

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim)) 
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))    
        self.glu1 = nn.Linear(self.dim, self.dim)       
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)   
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()   
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)  
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)  

        self.reset_parameters()   

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
  
    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)
        
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        # inputs = self.embedding(inputs)
        # targets =self.embedding(targets)
        pos_emb = self.pos_embedding.weight[:len]   
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)   

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)  
        hs = hs.unsqueeze(-2).repeat(1, len, 1)  
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)    
        # hs_expanded = hs.unsqueeze(1).expand(-1, nh.size(1), -1)
        ns = torch.sigmoid(self.glu1(nh) + self.glu2(hs)) 
        beta = torch.matmul(ns, self.w_2)    
        beta = beta * mask   
        select = torch.sum(beta * hidden, 1) 

        
        neighbor_sess = self.FindNeighbor(select)

        select = select + neighbor_sess

        b = self.embedding.weight[1:]  
        scores = torch.matmul(select, b.transpose(1, 0))   
        return scores

    def SSL(self, h_local, h_global):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)
        pos = score(h_local, h_global)
        neg1 = score(h_global, row_column_shuffle(h_local))
        one = torch.cuda.FloatTensor(neg1.shape).fill_(1)
        # one = torch.FloatTensor(neg1.shape).fill_(1)
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        return con_loss



    def forward(self, inputs, adj, mask_item, item, targets):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs) 
        targets1 =self.embedding(targets) 
        h_local = self.local_agg(h, adj, mask_item)
        item_neighbors = [inputs] 
        weight_neighbors = []   
        support_size = seqs_len
        

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]   
        weight_vectors = weight_neighbors  

        session_info = []   
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1) 
        
        # mean 
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1) 
        
        # sum
        # sum_item_emb = torch.sum(item_emb, 1)
        
        sum_item_emb = sum_item_emb.unsqueeze(-2)   


        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))
     
        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]  
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop+1].view(shape),  
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num), 
                                    extra_vector=session_info[hop]) 
                entity_vectors_next_iter.append(vector)  
            entity_vectors = entity_vectors_next_iter  

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim) 

        # combine
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)   
        h_global = F.dropout(h_global, self.dropout_global, training=self.training) 
        # h_target = F.dropout(h_target, self.dropout_target, training=self.training)
        output = h_local + h_global  
        # output = torch.cat((h_target, output), dim=1)
        con_loss = self.SSL(h_local, h_global)
        return output, self.beta*con_loss



def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    targets = trans_to_cuda(targets).long()
    

    hidden, con_loss = model(items, adj, mask, inputs, targets)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    # hs, con_loss = model(inputs, adj, mask, items)
    return targets, model.compute_scores(seq_hidden, mask), con_loss



def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()  
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores, con_loss = forward(model,  data)  
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss = loss + con_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss  
    print('\ttotal_Loss:\t%.3f' % total_loss)   
    model.scheduler.step()  
    
    

    print('start predicting: ', datetime.datetime.now())  
    model.eval()   
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []   
    hit, mrr = [], []
    for data in test_loader:
        targets, scores, con_loss = forward(model, data)
        sub_scores = scores.topk(20)[1] 
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()  
        # targets = targets.numpy() 
        targets = targets.cpu().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)

    return result
