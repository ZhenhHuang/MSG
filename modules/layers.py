import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from modules.neuron import RiemannianIFNode


class RSEncoderLayer(nn.Module):
    def __init__(self, manifold, T, in_dim, out_dim, v_threshold=1.):
        super(RSEncoderLayer, self).__init__()
        self.manifold = manifold
        self.fc = GCNConv(in_dim, out_dim)
        self.T = T
        self.drop = nn.Dropout(0.0)
        self.neuron = RiemannianIFNode(manifold, v_threshold)

    def forward(self, x, edge_index):
        x = self.drop(self.fc(x, edge_index))
        z = self.manifold.origin(x.shape, device=x.device, dtype=x.dtype)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1)
        o_seq, z_seq = self.neuron(x_seq, x, z)
        return o_seq, z_seq


class RiemannianSGNNLayer(nn.Module):
    def __init__(self, manifold, channels, v_threshold=1.0):
        super(RiemannianSGNNLayer, self).__init__()
        self.manifold = manifold
        self.layer = GCNConv(channels, channels, bias=False)
        self.neuron = RiemannianIFNode(manifold, v_threshold)
        self.drop = nn.Dropout(0.0)

    def forward(self, s_seq, z_seq, edge_index):
        """

        :param s_seq: [T, N, D]
        :param edge_index: [E, ]
        :return:
        """
        # print(z_seq.max())
        x_seq = self.drop(self.layer(s_seq, edge_index))
        y_seq = x_seq.mean(0)
        o_seq, z_seq = self.neuron(x_seq, y_seq, z_seq)
        # print(x_seq.max(), self.manifold.norm(y_seq).abs().max(), z_seq.max())
        return o_seq, z_seq