import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import dropout_edge
from modules.neuron import RiemannianNeuron, Neuron


class DropEdge(nn.Module):
    def __init__(self, p):
        super(DropEdge, self).__init__()
        self.p = p

    def forward(self, edge_index):
        return dropout_edge(edge_index, self.p, training=self.training)[0]


class RSEncoderLayer(nn.Module):
    def __init__(self, manifold, T, in_dim, out_dim, neuron,
                 v_threshold=1., delta=0.05, tau=2.,
                 step_size=0.1, dropout=0.0, use_MS=True):
        super(RSEncoderLayer, self).__init__()
        self.manifold = manifold
        self.fc = GCNConv(in_dim, out_dim, bias=False)
        self.T = T
        self.drop = nn.Dropout(dropout)
        self.drop_edge = DropEdge(dropout)
        self.neuron = RiemannianNeuron[neuron](manifold, v_threshold, delta, tau) if use_MS \
            else Neuron[neuron](manifold, v_threshold, delta, tau)
        self.step_size = step_size

    def forward(self, x, edge_index):
        edge_index = self.drop_edge(edge_index)
        x = self.drop(self.fc(x, edge_index))
        z = self.manifold.origin(x.shape, device=x.device, dtype=x.dtype)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1)
        o_seq, z_seq = self.neuron(x_seq, x * self.step_size, z)
        return o_seq, z_seq


class RiemannianSGNNLayer(nn.Module):
    def __init__(self, manifold, channels, neuron, v_threshold=1.0,
                 delta=0.05, tau=2.0, step_size=0.1, dropout=0.0, use_MS=True):
        super(RiemannianSGNNLayer, self).__init__()
        self.manifold = manifold
        self.layer = GCNConv(channels, channels, bias=False)
        self.neuron = RiemannianNeuron[neuron](manifold, v_threshold, delta, tau) if use_MS \
            else Neuron[neuron](manifold, v_threshold, delta, tau)
        self.drop = nn.Dropout(dropout)
        self.drop_edge = DropEdge(dropout)
        self.step_size = step_size

    def forward(self, s_seq, z_seq, edge_index):
        """

        :param s_seq: [T, N, D]
        :param edge_index: [E, ]
        :return:
        """
        # print(z_seq.max())
        edge_index = self.drop_edge(edge_index)
        x_seq = self.drop(self.layer(s_seq, edge_index))
        y_seq = x_seq.mean(0) * self.step_size
        o_seq, z_seq = self.neuron(x_seq, y_seq, z_seq)
        # print(x_seq.max(), self.manifold.norm(y_seq).abs().max(), z_seq.max())
        return o_seq, z_seq