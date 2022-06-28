import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Aggregators1 import Social_Aggregator1
from Social_Encoders1 import Social_Encoder1
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
from Location_Encoders import Location_Encoder
import math
#k-means=10,social=1
"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""

test1_u = []
test1_v = []
class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_u_history, enc_v_history, location_l, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim
        self.enc_u_history = enc_u_history
        self.location_l = location_l

        # self.w_ur0 = nn.Linear(self.embed_dim*2, self.embed_dim*2)
        self.w_ur1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn0 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u_s = self.enc_u(nodes_u)
        embeds_u_h = self.enc_u_history(nodes_u)
        embeds_v_h = self.enc_v_history(nodes_v)
        embeds_v_l = self.location_l(nodes_v)
        embeds_v = torch.cat((embeds_v_l, embeds_v_h), 1)
        embeds_u = torch.cat((embeds_u_s, embeds_u_h), 1)
        #  print(len(embeds_u))
        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer):
    model.train()
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()  # 计算新的导数时，进行的清零的操作
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)  # 计算梯度
        optimizer.step()  # 进行单次优化，一旦梯度被如backward()之类的函数计算好后，我们就可以调用这个函数。
    return 0

def test(model, device):
    model.eval()
    tmp_pred1 = []
    with torch.no_grad():
        test2_u = torch.tensor(test1_u).to(device).long()
        test2_v = torch.tensor(test1_v).to(device).long()
        val_output1 = model.forward(test2_u, test2_v)
        tmp_pred1.append(list(val_output1.data.cpu().numpy()))
    tmp_pred1 = np.array(sum(tmp_pred1, []))
    return test2_u, test2_v, tmp_pred1

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=16, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    '''
    dir_data = 'dataset'
    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    '''
    with open('dataset1.pickle', 'rb') as f:
        history_u_lists = pickle.load(f)
        history_ur_lists = pickle.load(f)
        history_v_lists = pickle.load(f)
        history_vr_lists = pickle.load(f)
        train_u = pickle.load(f)
        train_v = pickle.load(f)
        train_r = pickle.load(f)
        test_u = pickle.load(f)
        test_v = pickle.load(f)
        test_r = pickle.load(f)
        social_adj_lists = pickle.load(f)
        True_u = list(set(test_u))
        True_v = list(set(test_v))
        uv=[[]for i in range(len(True_u))]
        train_uv=[[]for i in range(len(True_u))]
        test_uv=[[]for i in range(len(True_u))]
        for i in range(len(True_u)):
            uv[i] = [x for (x, y) in enumerate(train_u) if y == True_u[i]]
            for j in range(len(uv[i])):
                train_uv[i].append(train_v[uv[i][j]])
            test_uv[i] = set(test_v)-set(train_uv[i])
        for i in range(len(True_u)):
            for j in range(len(test_uv[i])):
                test1_u.append(True_u[i])
                test1_v.append(list(test_uv[i])[j])

    location = []
    location_list = {}
    a = []
    f = open('k-means-result-40.txt', 'r')
    f = f.readlines()
    for i in f:
        i = i.split('\n')[0]
        i = i.split(',')
        location.append(i[1])
        location_list[int(i[0])] = int(i[1])
    location_num = len(set(location))
    # print(location_list)

    """
    ## toy dataset 
    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)

    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)

    # please add the validation set

    social_adj_lists: user's connected neighborhoods
    ratings_list: rating value from 0.5 to 4.0 (8 opinion embeddings)
    """

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))  # 包装数据和目标张量的数据集。通过沿着第一个维度索引两个张量来恢复每个样本。
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True)  # 数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()

    num_ratings = 6
    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)
    l2e = nn.Embedding(location_num, embed_dim).to(device)
    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # neighobrs
    agg_u_social = Social_Aggregator1(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    enc_u = Social_Encoder1(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
                           base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)
    location_l = Location_Encoder(v2e, l2e, embed_dim, location_list, cuda=device)
    # model
    graphrec = GraphRec(enc_u, enc_u_history, enc_v_history, location_l, r2e).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    #for epoch in range(1, args.epochs + 1):
    for epoch in range(1, 10):
        train(graphrec, device, train_loader, optimizer)
    user_lis, poi_lis, rating_list = test(graphrec, device)
    user_list = np.array(user_lis)
    poi_list = np.array(poi_lis)
    L_u = list(set(user_list))
    Luv = [[] for row in range(len(L_u))]
    poi = [[] for row in range(len(L_u))]
    rat = [[] for row in range(len(L_u))]
    up_list = [[] for row in range(len(L_u))]
    up = [[] for row in range(len(L_u))]
    for i in range(len(L_u)):
        Luv[i] = [x for (x, y) in enumerate(user_list) if y == int(L_u[i])]
        for j in range(len(Luv[i])):
            poi[i].append(int(poi_list[Luv[i][j]]))
            rat[i].append(float(rating_list[Luv[i][j]]))
        up_list[i] = sorted(range(len(rat[i])), key=lambda k: rat[i][k], reverse=True)
        for j in range(5):
            up[i].append(poi[i][up_list[i][j]])
    loc_v = [[] for row in range(len(L_u))]
    right_v = [[] for row in range(len(L_u))]
    count = 0
    count_ndcg=0
    correct = 0
    sign = []
    Lsign = []
    dcg=0
    idcg=0
    ndcg = 0
    for i in range(len(L_u)):
        loc_v[i] = [x for (x, y) in enumerate(test_u) if y == int(L_u[i])]
        for j in range(len(loc_v[i])):
            right_v[i].append(int(test_v[loc_v[i][j]]))
        if (len(right_v[i])) >= 5:
            Lright = list(set(up[i]) & set(right_v[i]))
            correct0 = len(Lright) / 5.0
            correct = correct + correct0
            if correct0 > 0:
                for j in range(len(Lright)):
                    sign.append(up[i].index(Lright[j]))
                for k in range(5):
                    if k in sign:
                        Lsign.append(1)
                    else:
                        Lsign.append(0)
                    dcg = dcg + (math.pow(2, Lsign[k]) - 1) / math.log2(k + 2)
                for k in range(len(Lright)):
                    idcg = idcg + (math.pow(2, 1) - 1) / math.log2(k + 2)
                ndcg0 = dcg / idcg
                sign.clear()
                Lsign.clear()
                ndcg = ndcg + ndcg0
                dcg = 0
                idcg = 0
                count_ndcg+=1
            count += 1
    if count == 0 or count_ndcg == 0:
        precision = 0
        ndcg = 0
    else:
        precision = correct / count
        ndcg = ndcg / count_ndcg
    print(count, "the precision is", precision," the NDCG is", ndcg)



if __name__ == "__main__":
    main()
