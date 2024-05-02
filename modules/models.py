import torch
import torch.nn as nn
from modules.spike_encoder import GNNSpikeEncoder, PoissonSpikeEncoder
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from modules.layers import RiemannianSGNNLayer, RSEncoderLayer
from manifolds.lorentz import Lorentz


class RiemannianSpikeGNN(nn.Module):
    def __init__(self, manifold, T, n_layers, in_dim, embed_dim):
        super(RiemannianSpikeGNN, self).__init__()
        if isinstance(manifold, Lorentz):
            embed_dim += 1
        self.manifold = manifold
        self.encoder = RSEncoderLayer(manifold, T, in_dim, embed_dim)
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(RiemannianSGNNLayer(manifold, embed_dim))

    def forward(self, data):
        x, edge_index = data['features'], data['edge_index']
        x, z = self.encoder(x, edge_index)
        for layer in self.layers:
            x, z = layer(x, z, edge_index)
        if isinstance(self.manifold, Lorentz):
            z = self.manifold.to_poincare(z)
        return z


class SpikeClassifier(nn.Module):
    def __init__(self, T, backbone, n_layers, n_neurons, hidden_neurons,
                 embed_neurons, num_classes, n_heads, dropout):
        super(SpikeClassifier, self).__init__()
        self.encoder = GNNSpikeEncoder(T, backbone, n_layers, n_neurons, hidden_neurons, embed_neurons, n_heads,
                                       dropout)
        self.fc = nn.Linear(embed_neurons, embed_neurons)
        self.lif = MultiStepLIFNode(detach_reset=True)
        self.proj = nn.Linear(embed_neurons, num_classes)
        self.T = T

    def forward(self, data):
        x, edge_index = data['features'], data['edge_index']
        x = self.encoder(x, edge_index)  # [T, N, D]
        x = self.lif(self.fc(x))    # [T, N, C]
        x = self.proj(x)
        return x.sum(0) / self.T


class SpikeLinkPredictor(nn.Module):
    def __init__(self, T, backbone, n_layers, n_neurons, hidden_neurons,
                 embed_neurons, n_heads, dropout):
        super().__init__()
        self.encoder = GNNSpikeEncoder(T, backbone, n_layers, n_neurons, hidden_neurons, embed_neurons, n_heads,
                                       dropout, v_threshold=0.1)
        self.fc = nn.Linear(embed_neurons, embed_neurons)
        self.lif = MultiStepLIFNode(detach_reset=True, v_threshold=0.1)
        self.proj = nn.Linear(embed_neurons, embed_neurons)
        self.T = T

    def forward(self, data):
        x, edge_index = data['features'], data[f'pos_edges_train']
        x = self.encoder(x, edge_index)  # [T, N, D]
        x = self.lif(self.fc(x))  # [T, N, D]
        x = self.proj(x)    # [T, N, d]
        return x.mean(0)


class FermiDiracDecoder(nn.Module):
    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = torch.sigmoid((self.r - dist) / self.t)
        return probs


if __name__ == '__main__':
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='../data', name='Cora')
    encoder = RiemannianSpikeGNN(Lorentz(), 10, 2, 1433, 32)
    spike_sequence = encoder(dataset.data.x, dataset.data.edge_index)