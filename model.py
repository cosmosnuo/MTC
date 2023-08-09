import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv as GCN
from torch.nn.parameter import Parameter


class GraphEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, n_clusters):
        super(GraphEmbedding, self).__init__()
        # nn.Linear()
        self.weight1 = nn.Parameter(torch.randn(input_size, input_size))
        self.weight2 = nn.Parameter(torch.randn(input_size, input_size))
        self.weight3 = nn.Parameter(torch.randn(input_size, input_size))
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, x1, x2, x3):
        y1 = torch.matmul(x1, self.weight1)
        y2 = torch.matmul(x2, self.weight2)
        y3 = torch.matmul(x3, self.weight3)
        y = y1 + y2 + y3
        y = self.fc(y)
        return y


class GAE(nn.Module):
    def __init__(self, in_dim, hidden_dims_v, hidden_dims, views, v=1):
        super(GAE, self).__init__()
        self.views = views

        # layer: v0
        layers0 = [GCN(in_feats=in_dim[0], out_feats=hidden_dims_v[0], activation=F.relu)]
        layers0.append(GCN(in_feats=hidden_dims_v[0], out_feats=hidden_dims_v[1], activation=lambda x: x))

        # layer: v1
        layers1 = [GCN(in_feats=in_dim[1], out_feats=hidden_dims_v[0], activation=F.relu)]
        layers1.append(GCN(in_feats=hidden_dims_v[0], out_feats=hidden_dims_v[1], activation=lambda x: x))

        # layer: v2
        layers2 = [GCN(in_feats=in_dim[2], out_feats=hidden_dims_v[0], activation=F.relu)]
        layers2.append(GCN(in_feats=hidden_dims_v[0], out_feats=hidden_dims_v[1], activation=lambda x: x))

        # layer: vm
        layers_m = [GCN(in_feats=hidden_dims_v[1], out_feats=hidden_dims[0], activation=F.relu)]
        layers_m.append(GCN(in_feats=hidden_dims[0], out_feats=hidden_dims[1], activation=lambda x: x))

        self.layer = nn.ModuleList(layers_m)
        self.layer0 = nn.ModuleList(layers0)
        self.layer1 = nn.ModuleList(layers1)
        self.layer2 = nn.ModuleList(layers2)
        self.featfusion = FeatureFusion(size=hidden_dims_v[1])
        self.decoder = InnerProductDecoder(size=hidden_dims[1])

    def forward(self, graph0, graph1, graph2, feature0, feature1, feature2, graph):
        h0 = feature0
        h1 = feature1
        h2 = feature2

        for conv in self.layer0:
            h0 = conv(graph0, h0)
        for conv in self.layer1:
            h1 = conv(graph1, h1)
        for conv in self.layer2:
            h2 = conv(graph2, h2)
        xh = self.featfusion(h0, h1, h2)
        xh0 = xh
        for conv in self.layer:
            xh = conv(dgl.add_self_loop(graph), xh)
        adj_rec = {}
        for i in range(self.views):
            adj_rec[i] = self.decoder(xh)

        return adj_rec, xh


class FeatureFusion(nn.Module):
    def __init__(self, size):
        super(FeatureFusion, self).__init__()
        self.weight1 = nn.Parameter(torch.randn(size, size))
        self.weight2 = nn.Parameter(torch.randn(size, size))
        self.weight3 = nn.Parameter(torch.randn(size, size))
        self.fc = nn.Linear(size, size)

    def forward(self, x1, x2, x3):

        y1 = torch.matmul(x1, self.weight1)
        y2 = torch.matmul(x2, self.weight2)
        y3 = torch.matmul(x3, self.weight3)
        y = y1 + y2 + y3
        y = self.fc(y)
        return y


class InnerProductDecoder(nn.Module):
    def __init__(self, size, activation=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.activation = activation
        self.weight = Parameter(torch.FloatTensor(size, size))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, z):
        t = torch.mm(z, self.weight)
        adj = self.activation(torch.mm(t, z.t()))
        return adj
