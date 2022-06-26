import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from Attention import Attention

class Social_Aggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, features, u2e, embed_dim, cuda="cpu"):
        super(Social_Aggregator, self).__init__()

        self.features = features
        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, to_neighs,to_neighs_1):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)
            embed_matrix_0=torch.empty(num_neighs, self.embed_dim, dtype=torch.float).to(self.device)
            for j in range(num_neighs):
               # tmp_adj=list(tmp_adj)
               # a=list(tmp_adj[j])
                b=to_neighs_1[i][j]
                num_neighs_0=len(b)
                c=self.u2e.weight[list(b)]
                d=self.u2e.weight[tmp_adj[j]]
                att_w_0 = self.att(c, d, num_neighs_0)
                att_history_0 = torch.mm(c.t(), att_w_0).t()
                embed_matrix_0[j] = att_history_0
            e_u = embed_matrix_0
            #e_u = self.u2e.weight[list(tmp_adj)] # fast: user embedding
            #slow: item-space user latent factor (item aggregation)
            #feature_neigbhors = self.features(torch.LongTensor(list(tmp_adj)).to(self.device))
            #e_u = torch.t(feature_neigbhors)
            u_rep = self.u2e.weight[nodes[i]]
            att_w = self.att(e_u, u_rep, num_neighs)
            att_history = torch.mm(e_u.t(), att_w).t()
            embed_matrix[i] = att_history
        to_feats = embed_matrix
        return to_feats


