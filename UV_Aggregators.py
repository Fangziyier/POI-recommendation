import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention

class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.v2e = v2e#item的潜入向量
        self.r2e = r2e#评分的嵌入向量
        self.u2e = u2e#用户的嵌入向量
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)
       # print(self.r2e)
    def forward(self, nodes, history_uv, history_r):
       # print(self.v2e)
        embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)#dtype返回张量的数据类型为float型
        #返回填充有未初始化数据的张量，即
        for i in range(len(history_uv)):
            history = history_uv[i]#一个用户交互过的所有item,为一个数组，eg.[1,2,5,4]
            num_histroy_item = len(history)#交互过的项目数量
            tmp_label = history_r[i]
          #  print(tmp_label)#一个用户交互过的所有item对应的评分，为一个数组，eg.[1,2,5,4]
            if self.uv == True:
                # user component
                e_uv = self.v2e.weight[history]# e_uv为一个用户交互过的所有item的嵌入向量
                uv_rep = self.u2e.weight[nodes[i]]#为一个用户的嵌入向量
            else:
                # item component
                e_uv = self.u2e.weight[history]
                uv_rep = self.v2e.weight[nodes[i]]
            e_r = self.r2e.weight[tmp_label]#一个用户交互过的所有item对应的评分，为一个二维嵌入向量，eg.[1,2,5,4]
            x = torch.cat((e_uv, e_r), 1)#连接用户交互过的item嵌入向量和评分向量这两个向量
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))#o_history为公式（2）中的xia

            att_w = self.att(o_history, uv_rep, num_histroy_item)
            att_history = torch.mm(o_history.t(), att_w)#o_history.t()和att_w两个矩阵相乘
            att_history = att_history.t()

            embed_matrix[i] = att_history
        to_feats = embed_matrix
        return to_feats
