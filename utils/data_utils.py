import torch
import numpy as np
import torch_geometric.data
from torch_geometric.datasets import Amazon, Coauthor
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomNodeSplit
import warnings

warnings.filterwarnings('ignore')


def load_data(root: str, data_name: str, split='public', num_val=0.1, num_test=0.8):
    if data_name in ["computers", "photo"]:
        dataset = Amazon(root, name=data_name)
        dataset.data = RandomNodeSplit(num_val=num_val, num_test=num_test)(dataset[0])
        train_mask, val_mask, test_mask = dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask
    elif data_name == "KarateClub":
        dataset = KarateClub()
        dataset.data = RandomNodeSplit(num_val=0.2, num_test=0.3)(dataset.data[0])
        train_mask, val_mask, test_mask = dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask
    elif data_name in ["CS", "Physics"]:
        dataset = Coauthor(root, name=data_name)
        dataset.data = RandomNodeSplit(num_val=num_val, num_test=num_test)(dataset[0])
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
