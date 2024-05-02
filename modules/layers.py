import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from modules.neuron import RiemannianIFNode


class RSEncoderLayer(nn.Module):
    def __init__(self, manifold, T, in_dim, out_dim):
        super(RSEncoderLayer, self).__init__()
        self.manifold = manifold
        self.fc = GCNConv(in_dim, out_dim)
        self.T = T

    def forward(self, x, edge_index):
        x = self.fc(x, edge_index)
        z = self.manifold.expmap0(x)
        x = x.unsqueeze(0).repeat(self.T, 1, 1)
        return x, z


class RiemannianSGNNLayer(nn.Module):
    def __init__(self, manifold, channels, v_threshold=1.0):
        super(RiemannianSGNNLayer, self).__init__()
        self.manifold = manifold
        self.layer = GCNConv(channels, channels, bias=False)
        self.neuron = RiemannianIFNode(manifold, v_threshold)

    def forward(self, s_seq, z_seq, edge_index):
        """

        :param s_seq: [T, N, D]
        :param edge_index: [E, ]
        :return:
        """
        x_seq = self.layer(s_seq, edge_index)
        y_seq = self.layer.lin(x_seq.mean(0))
        o_seq, z_seq = self.neuron(x_seq, y_seq, z_seq)
        return o_seq, z_seq