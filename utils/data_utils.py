import random

import torch
import networkx as nx
import numpy as np
import torch_geometric.data
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, GemsecDeezer, WikiCS, FacebookPagePage, Amazon
from torch_geometric.utils import to_networkx
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomNodeSplit
# from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.datasets import load_wine, load_breast_cancer, load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import pickle as pkl
import os
import warnings

warnings.filterwarnings('ignore')


def get_mask(idx, length):
    """Create mask.
    """
    mask = torch.zeros(length, dtype=torch.bool)
    mask[idx] = 1
    return mask


def load_data(root: str, data_name: str, split='public', num_val=0.1, num_test=0.8):
    if data_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=root, name=data_name, split=split)
        train_mask, val_mask, test_mask = dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask
    elif data_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=data_name)
        mask = dataset.get_idx_split()
        train_mask, val_mask, test_mask = mask.values()
    elif data_name in ['actor', 'chameleon', 'squirrel']:
        if data_name == 'actor':
            path = root + f'/{data_name}'
            dataset = Actor(root=path)
        else:
            dataset = WikipediaNetwork(root=root, name=data_name)
        num_nodes = dataset.data.x.shape[0]
        idx_train = []
        for j in range(dataset.num_classes):
            idx_train.extend([i for i, x in enumerate(dataset.data.y) if x == j][:20])
        idx_val = np.arange(num_nodes - 1500, num_nodes - 1000)
        idx_test = np.arange(num_nodes - 1000, num_nodes)
        label_len = dataset.data.y.shape[0]
        train_mask, val_mask, test_mask = get_mask(idx_train, label_len), get_mask(idx_val, label_len), get_mask(
            idx_test, label_len)
    elif data_name == "airport":
        dataset = Airport(root)
        train_mask, val_mask, test_mask = dataset.data.mask
    elif data_name in ["computers", "photo"]:
        dataset = Amazon(root, name=data_name)
        dataset.data = RandomNodeSplit(num_val=num_val, num_test=num_test)(dataset[0])
        train_mask, val_mask, test_mask = dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask
    elif data_name == "wikics":
        dataset = WikiCS(root=f"{root}/wikics")
        train_mask, val_mask, test_mask = dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask
    elif data_name == "facebook":
        dataset = FacebookPagePage(root=f"{root}/facebook")
        n = len(dataset.data.x)
        index = torch.arange(n)
        train_mask = index[: int(n * 0.7)]
        val_mask = index[int(n * 0.7): int(n * 0.8)]
        test_mask = index[int(n * 0.8):]
    elif data_name == "KarateClub":
        dataset = KarateClub()
        dataset.data = RandomNodeSplit(num_val=0.2, num_test=0.3)(dataset.data[0])
        train_mask, val_mask, test_mask = dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask
    else:
        raise NotImplementedError

    print(dataset.data)
    features = dataset.data.x
    num_features = dataset.num_features
    labels = dataset.data.y
    edge_index = dataset.data.edge_index.long()
    neg_edges = negative_sampling(edge_index)
    num_classes = dataset.num_classes

    pos_edges, neg_edges = mask_edges(edge_index, neg_edges, 0.05, 0.1)

    data = {
        "features": features,
        "num_features": num_features,
        "labels": labels,
        "num_classes": num_classes,
        "edge_index": edge_index,
        "pos_edges_train": pos_edges[0],
        "pos_edges_val": pos_edges[1],
        "pos_edges_test": pos_edges[2],
        "neg_edges_train": neg_edges[0],
        "neg_edges_val": neg_edges[1],
        "neg_edges_test": neg_edges[2],
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask
    }

    return data


def mask_edges(edge_index, neg_edges, val_prop, test_prop):
    n = len(edge_index[0])
    n_val = int(val_prop * n)
    n_test = int(test_prop * n)
    edge_val, edge_test, edge_train = edge_index[:, :n_val], edge_index[:, n_val:n_val + n_test], edge_index[:,
                                                                                                  n_val + n_test:]
    val_edges_neg, test_edges_neg = neg_edges[:, :n_val], neg_edges[:, n_val:n_test + n_val]
    train_edges_neg = torch.concat([neg_edges, val_edges_neg, test_edges_neg], dim=-1)
    return (edge_train, edge_val, edge_test), (train_edges_neg, val_edges_neg, test_edges_neg)


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.shape[0], 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


class Airport:
    def __init__(self, root):
        super(Airport, self).__init__()
        val_prop, test_prop = 0.15, 0.15
        graph = pkl.load(open(f"{root}/airport/airport.p", 'rb'))
        adj = nx.adjacency_matrix(graph).toarray()
        row, col = np.nonzero(adj)
        edge_index = np.concatenate([row[None], col[None]], axis=0)
        features = np.array([graph._node[u]['feat'] for u in graph.nodes()])
        features = augment(adj, torch.tensor(features).float())
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])

        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, random.seed(3047))
        mask = (idx_train, idx_val, idx_test)

        self.data = torch_geometric.data.Data(x=features,
                                              edge_index=torch.tensor(edge_index),
                                              y=torch.tensor(labels),
                                              mask=mask)


class KarateClub:
    def __init__(self):
        data = torch_geometric.datasets.KarateClub()
        self.feature = data.x
        self.num_features = data.x.shape[1]
        self.num_nodes = data.x.shape[0]
        self.edge_index = data.edge_index
        self.weight = torch.ones(self.edge_index.shape[1])
        self.labels = data.y
        self.num_classes = len(np.unique(self.labels))
        self.neg_edge_index = negative_sampling(data.edge_index)
        self.data = data
