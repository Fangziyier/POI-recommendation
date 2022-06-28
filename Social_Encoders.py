import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Social_Encoder(nn.Module):

    def __init__(self, features, embed_dim, social_adj_lists, aggregator, base_model=None, cuda="cpu"):
        super(Social_Encoder, self).__init__()

        self.features = features
        self.social_adj_lists = social_adj_lists
        self.aggregator = aggregator
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #

    def forward(self, nodes):

        to_neighs = []
        to_neighs_0=[]
        to_neighs_1=[]
        for node in nodes:
            t=0
            to_neighs.append(self.social_adj_lists[int(node)])
            for j in list(self.social_adj_lists[int(node)]):
                to_neighs_0.append(self.social_adj_lists[int(j)])
            to_neighs_1.append(to_neighs_0)
            to_neighs_0=[]

        neigh_feats = self.aggregator.forward(nodes, to_neighs, to_neighs_1)  # user-user network

        self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
        self_feats = self_feats.t()

        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
