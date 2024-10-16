import torch
import torch.nn as nn
from modules.layers import RiemannianSGNNLayer, RSEncoderLayer
from manifolds.lorentz import Lorentz
from geoopt.tensor import ManifoldParameter
import math


class RiemannianSpikeGNN(nn.Module):
    def __init__(self, manifold, T, n_layers, step_size, in_dim, embed_dim,
                 n_classes, v_threshold=1.0, dropout=0.1, neuron="IF", delta=0.05, tau=2,
                task='NC', use_MS=True, device=None):
        super(RiemannianSpikeGNN, self).__init__()
        if isinstance(manifold, Lorentz):
            embed_dim += 1
        self.manifold = manifold
        self.step_size = step_size
        self.task = task
        self.encoder = RSEncoderLayer(manifold, T, in_dim, embed_dim, neuron=neuron, delta=delta, tau=tau,
                                      step_size=step_size, v_threshold=v_threshold,
                                      dropout=dropout, use_MS=use_MS)
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(
                RiemannianSGNNLayer(manifold, embed_dim, neuron=neuron, delta=delta, tau=tau,
                                                       step_size=step_size, v_threshold=v_threshold,
                                                       dropout=dropout, use_MS=use_MS)
                                   )
        self.fc = nn.Linear(embed_dim, n_classes, bias=False) if task == "NC" else None

    def forward(self, data):
        x = data['features']
        if self.task == 'NC':
            edge_index = data['edge_index']
        elif self.task == 'LP':
            edge_index = data['pos_edges_train']
        else:
            raise NotImplementedError
        x, z, y = self.encoder(x, edge_index)
        v = y.clone()
        for layer in self.layers:
            x, z, y = layer(x, z, edge_index)
            v += y

        if self.task == 'NC':
            z1 = self.manifold.proju0(self.manifold.logmap0(z))
            z1 = self.fc(z1)
            return z1, self.fc(self.manifold.proju0(v))
        elif self.task == "LP":
            return self.manifold.proju0(self.manifold.logmap0(z)), self.manifold.proju0(v)


class FermiDiracDecoder(nn.Module):
    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = torch.sigmoid((self.r - dist) / self.t)
        return probs