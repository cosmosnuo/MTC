import argparse
import os
import dgl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors
from time import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import linalg as LA
from sklearn.preprocessing import normalize

from model import GraphEmbedding, GAE

parser = argparse.ArgumentParser(description='GAE')
parser.add_argument('-k', type=int, default=3, help='number of k in k-nn')
parser.add_argument('-ptrain_epochs', '-pe', type=int, default=2000, help='number of pre-train_epochs')
parser.add_argument('-train_epochs', '-te', type=int, default=400, help='number of train_epochs')
parser.add_argument('-hidden_dimsV', '-hv',type=int, nargs='+', default=[64, 64], help='list of V1 hidden dimensions')
parser.add_argument('-hidden_dims', '-hd', type=int, nargs='+', default=[64, 64], help='list of feature hidden dimensions')
parser.add_argument('-plr', type=float, default=0.0001, help='Adam learning rate')
parser.add_argument('-tlr', type=float, default=0.01, help='Adam learning rate')
parser.add_argument('-lambda1', '-la1', type=float, default=0.001, help='Rate for gtr')
parser.add_argument('-lambda2', '-la2', type=float, default=1, help='Rate for clu')
parser.add_argument('-dataset', type=str, default='overview', help='choose a dataset')
parser.add_argument('-datafile', '-df', type=str, default='./hospital_overview.npz', help='file of dataset')
parser.add_argument('-n_clusters', '-n', type=int, default=4, help='the final number of target classes')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print("use cuda: {}".format(args.cuda))
device = torch.device("cuda:0" if args.cuda else "cpu")
print(args)


def load_data(args):
    if args.dataset == 'overview':
        data = Dataset(name=args.dataset, k=args.k, datafile=args.datafile, n_clusters=args.n_clusters)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    return data


class Dataset(object):
    def __init__(self, n_clusters, name, k, datafile):
        self.n_clusters = n_clusters
        self.name = name
        self.k = k
        self.datafile = datafile

        F = np.load(self.datafile, allow_pickle=True)
        self.data = F['arr_0']
        self.feature = self.data
        print("视角", self.feature.shape)

        for i in range(self.feature.shape[0]):
            self.feature[i] = self.feature[i].T

        self.idx = np.arange(self.feature[0].shape[0])
        print(len(self.idx))
        self.graph_dict = {}

        for i in range(self.feature.shape[0]):
            g = pair(self.k, self.feature[i])
            self.graph_dict[i] = g

        for i in range(self.feature.shape[0]):
            self._load(self.feature[i], self.idx, self.graph_dict[i], i)

    def _load(self, feature, idx, graph, i):
        features = sp.csr_matrix(feature, dtype=np.float32)

        idx = np.asarray(idx, dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = graph
        edges = np.asarray(list(map(idx_map.get, edges_unordered.flatten())),
                           dtype=np.int32).reshape(edges_unordered.shape)

        adj = sp.coo_matrix((np.ones(edges.shape[0]),
                             (edges[:, 0], edges[:, 1])),
                            shape=(feature.shape[0], feature.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        self.graph_dict[i] = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
        # features = _normalize(features)
        self.feature[i] = np.asarray(features.todense())


def _normalize(mx):
    rowsum = np.asarray(mx.sum(1))
    rowsum[rowsum != 0] = 1.0 / rowsum[rowsum != 0]
    r_inv = rowsum.flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def pair(knn='', data='', metrix='euclidean'):
    x_train = data
    n_train = len(x_train)
    x_train_flat = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))[:n_train]
    train_neighbors = NearestNeighbors(n_neighbors=knn + 1, metric=metrix).fit(x_train_flat)
    _, idx = train_neighbors.kneighbors(x_train_flat)


    new_idx = np.empty((idx.shape[0], idx.shape[1] - 1))
    assert (idx >= 0).all()
    for i in range(idx.shape[0]):
        try:
            new_idx[i] = idx[i, idx[i] != i][:idx.shape[1] - 1]
        except Exception as e:
            print(idx[i, ...], new_idx.shape, idx.shape)
            raise e

    idx = new_idx.astype(int)
    graph = np.empty(shape=[0, 2], dtype=int)
    for i, m in enumerate(idx):
        for mm in m:
            graph = np.append(graph, [[i, mm]], axis=0)
    return graph


def load_dataset():
    data = load_data(args)
    views = len(data.feature)
    feature = data.feature
    n_clusters = data.n_clusters
    print('Number of Samples: {:02d}'.format(data.feature[0].shape[0]))
    print('Views:', views)


    feature0 = torch.FloatTensor(data.feature[0])
    feature1 = torch.FloatTensor(data.feature[1])
    feature2 = torch.FloatTensor(data.feature[2])

    print('VIEW-0 DateSize: {:02d} * {:02d} '.format(feature0.shape[0], feature0.shape[1]))
    print('VIEW-1 DateSize: {:02d} * {:02d} '.format(feature1.shape[0], feature1.shape[1]))
    print('VIEW-2 DateSize: {:02d} * {:02d} '.format(feature2.shape[0], feature2.shape[1]))

    in_feats = [feature0.shape[1], feature1.shape[1], feature2.shape[1]]

    graph0 = dgl.from_networkx(data.graph_dict[0])
    graph1 = dgl.from_networkx(data.graph_dict[1])
    graph2 = dgl.from_networkx(data.graph_dict[2])

    print('VIEW-0 Edges: {:02d} '.format(graph0.number_of_edges()))
    print('VIEW-1 Edges: {:02d} '.format(graph1.number_of_edges()))
    print('VIEW-2 Edges: {:02d} '.format(graph2.number_of_edges()))


    degs0 = graph0.in_degrees().float()
    degs1 = graph1.in_degrees().float()
    degs2 = graph2.in_degrees().float()

    norm0 = torch.pow(degs0, -0.5)
    norm1 = torch.pow(degs1, -0.5)
    norm2 = torch.pow(degs2, -0.5)

    norm0[torch.isinf(norm0)] = 0
    norm1[torch.isinf(norm1)] = 0
    norm2[torch.isinf(norm2)] = 0

    graph0.ndata['norm'] = norm0.unsqueeze(1)
    graph1.ndata['norm'] = norm1.unsqueeze(1)
    graph2.ndata['norm'] = norm2.unsqueeze(1)

    return in_feats, feature, feature0, feature1, feature2, graph0, graph1, graph2, views, n_clusters


def normalization(adj_r):
    adj_p = torch.clamp(adj_r, 0, 1)
    adj_p = torch.round(adj_p + 0.1)
    adj_pn = adj_p.detach().cpu().numpy()
    adj_pn += adj_pn.T
    graph = nx.from_numpy_array(adj_pn, create_using=nx.DiGraph())
    graph = dgl.from_networkx(graph)
    graph = graph.to(device)
    return graph


def solhouette(y_pred, feature):
    sil_avg = 0
    for i in range(feature.shape[0]):
        silhouette = silhouette_score(feature[i], y_pred)
        sil_avg = sil_avg + silhouette
        print('view_{:02d} :{:.5f}'.format(i, silhouette))

    print("sil_avg:", sil_avg/3)
    X = np.concatenate((feature[0], feature[1], feature[2]), axis=1)
    silhouette_avg = silhouette_score(X, y_pred)
    print("fusion:", silhouette_avg)


def trace_loss(adj, k):
    adj = torch.clamp(adj, 0, 1)
    adj = torch.round(adj)
    rowsum = adj.sum(axis=1).detach().cpu().numpy()
    d = torch.zeros(adj.shape).numpy()
    row, col = np.diag_indices_from(d)
    d[row, col] = rowsum
    l = d - adj.detach().cpu().numpy()
    e_vals, e_vecs = np.linalg.eig(l)
    sorted_indices = np.argsort(e_vals)
    q = torch.tensor(e_vecs[:, sorted_indices[0:k:]].astype(np.float32)).cuda()
    m = torch.mm(torch.t(q), adj)
    m = torch.mm(m, q)
    return torch.trace(m)


def shuffling(x, latent):
    idxs = torch.arange(0, x.shape[0]).cuda()
    a = torch.randperm(idxs.size(0)).cuda()
    aa = idxs[a].unsqueeze(1)
    aaa = aa.repeat(1, latent)
    return torch.gather(x, 0, aaa)


def main():

    in_feats, feature, feature0, feature1, feature2, graph0, graph1, graph2, views, n_clusters = load_dataset()
    print('Clusters:', n_clusters)

    # 获得共识图
    model_g = GraphEmbedding(feature0.shape[0], int(feature0.shape[0] / 2), n_clusters).cuda()
    optim_ge_p = torch.optim.Adam(model_g.parameters(), lr=args.plr)
    optim_ge_t = torch.optim.Adam(model_g.parameters(), lr=args.tlr)
    criterion_m = torch.nn.MSELoss()

    adj0 = graph0.adjacency_matrix().to_dense()
    adj1 = graph1.adjacency_matrix().to_dense()
    adj2 = graph2.adjacency_matrix().to_dense()

    edges = graph0.number_of_edges() + graph1.number_of_edges() + graph2.number_of_edges()


    criterion_m.cuda(device=device)
    model_g = model_g.to(device)

    graph0 = graph0.to(device)
    graph1 = graph1.to(device)
    graph2 = graph2.to(device)

    feature0 = feature0.to(device)
    feature1 = feature1.to(device)
    feature2 = feature2.to(device)

    adj0 = adj0.to(device)
    adj1 = adj1.to(device)
    adj2 = adj2.to(device)

    begin_time = time()
    print('GE Pre-Training Start')

    model_g.train()
    for epoch in range(args.ptrain_epochs):

        optim_ge_p.zero_grad()
        adj_r = model_g.forward(adj0, adj1, adj2)
        loss_ge = (criterion_m(adj_r, adj0) + criterion_m(adj_r, adj1) + criterion_m(adj_r,
                                                                                     adj2)) / views
        loss_ge.backward()
        optim_ge_p.step()

        if (epoch + 1) % 200 == 0:
            end_time = time()
            run_time = end_time - begin_time
            print(
                'GE-Pre-Training Epoch: {:02d} | GE-Loss: {:.5f} | Time: {:.2f}'.format(epoch + 1,
                                                                                        loss_ge,
                                                                                        run_time))

    graph = normalization(adj_r)

    model = GAE(in_feats, args.hidden_dimsV, args.hidden_dims, views).cuda()
    optim_gae_p = torch.optim.Adam(model.parameters(), lr=0.0001)
    optim_gae_t = torch.optim.Adam(model.parameters(), lr=0.0001)
    pos_weight = torch.Tensor([float(graph0.adjacency_matrix().to_dense().shape[0] ** 2 - edges / 2) / edges * 2])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = model.to(device)
    criterion.cuda(device=device)


    print('GAE Pre-Training Start')
    model.train()
    for epoch in range(args.ptrain_epochs):
        # train GAE
        optim_gae_p.zero_grad()
        adj_logits, z = model.forward(graph0, graph1, graph2, feature0, feature1, feature2, graph)
        loss_rec = (criterion(adj_logits[0], adj0) + criterion(adj_logits[1], adj1) + criterion(adj_logits[2],
                                                                                                adj2)) / views
        loss_rec.backward()
        optim_gae_p.step()

        if (epoch + 1) % 200 == 0:
            end_time = time()
            run_time = end_time - begin_time
            print('graph.number_of_edges', graph.number_of_edges())
            print(
                'GAE-Pre-Training Epoch: {:02d} | GAE-Loss: {:.5f} |  Time: {:.2f}'.format(epoch + 1,
                                                                                           loss_rec,
                                                                                           run_time))


    with torch.no_grad():
        _, z = model.forward(graph0, graph1, graph2, feature0, feature1, feature2, graph)
    kmeans = KMeans(n_clusters=n_clusters)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())

    print("...........Pre-training results............")
    print("...........z..............")
    solhouette(y_pred, feature)

    print('Training Start')
    for epoch in range(args.train_epochs):

        optim_ge_t.zero_grad()
        adj_r = model_g.forward(adj0, adj1, adj2)
        loss_gre = (criterion_m(adj_r, adj0) + criterion_m(adj_r, adj1) + criterion_m(adj_r, adj2)) / views
        loss_gtr = trace_loss(adj_r, n_clusters) ** 2
        loss_ge = loss_gre + args.lambda1 * loss_gtr
        loss_ge.backward()
        optim_ge_t.step()

        graph = normalization(adj_r)

        adj_logits, h = model.forward(graph0, graph1, graph2, feature0, feature1, feature2, graph)
        loss_rec = (criterion(adj_logits[0], adj0) + criterion(adj_logits[1], adj1) + criterion(adj_logits[2],
                                                                                                adj2)) / views
        loss_gae = loss_rec
        optim_gae_t.zero_grad()
        loss_gae.backward()
        optim_gae_t.step()

        if (epoch + 1) % 50 == 0:
            end_time = time()
            run_time = end_time - begin_time
            print(
                'Epoch: {:02d} | GAE-Loss: {:.5f}| GE-Loss: {:.5f} + {:.5f} =  {:.5f} | Time: {:.2f}'.format(
                    epoch + 1, loss_gae, loss_gre, args.lambda1 * loss_gtr, loss_ge, run_time))

    model.eval()
    _, z = model.forward(graph0, graph1, graph2, feature0, feature1, feature2, graph)

    kmeans = KMeans(n_clusters)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())

    print("..................finish........................")
    # print(y_pred)
    print(".................cluster...................")
    for i in range(n_clusters):
        class_i = np.where(y_pred == i)[0]
        print("class_{}:{}".format(i, class_i))

    print("...........Training results............")
    print("...........z..............")
    solhouette(y_pred, feature)

if __name__ == '__main__':
    main()