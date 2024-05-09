import torch
import torch.nn as nn
from modules.spike_encoder import GNNSpikeEncoder, PoissonSpikeEncoder
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from modules.layers import RiemannianSGNNLayer, RSEncoderLayer
from manifolds.lorentz import Lorentz
from manifolds.euclidean import Euclidean


class RiemannianSpikeGNN(nn.Module):
    def __init__(self, manifold, T, n_layers, step_size, in_dim, embed_dim,
                 n_classes, v_threshold=1.0, dropout=0.1, self_train=False, task='NC'):
        super(RiemannianSpikeGNN, self).__init__()
        if isinstance(manifold, Lorentz):
            embed_dim += 1
        self.manifold = manifold
        self.step_size = step_size
        self.self_train = self_train
        self.task = task
        self.encoder = RSEncoderLayer(manifold, T, in_dim, embed_dim,
                                      step_size=step_size, v_threshold=v_threshold, dropout=dropout)
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(
                RiemannianSGNNLayer(manifold, embed_dim,
                                    step_size=step_size, v_threshold=v_threshold, dropout=dropout)
            )
        self.fc = nn.Linear(embed_dim, n_classes) if task == "NC" else None

    def forward(self, data):
        x = data['features']
        if self.task == 'NC':
            edge_index = data['edge_index']
        elif self.task == 'LP':
            edge_index = data['pos_edges_train']
        else:
            raise NotImplementedError
        x, z = self.encoder(x, edge_index)
        for layer in self.layers:
            x, z = layer(x, z, edge_index)

        if self.task == 'NC' and self.self_train is False:
            z = self.manifold.proju0(self.manifold.logmap0(z))
            z = self.fc(z)
            return z
        elif self.task == "LP" and self.self_train is False:
            return z
        elif self.self_train:
            loss = self.self_loss(x.mean(0), z)
            return z, loss

    def self_loss(self, v, z, tau=0.2):
        v1 = v.unsqueeze(0)
        z1 = z.unsqueeze(1)
        dists = -self.manifold.dist(self.manifold.expmap(z1, v1), z1) / tau  # (N, N)
        dists = dists.exp()
        pos = dists.diag()
        prob = pos / dists.sum(-1)
        loss = -torch.log(prob.clamp(min=1e-8)).mean()
        return loss


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
        x = self.lif(self.fc(x))  # [T, N, C]
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
        x = self.proj(x)  # [T, N, d]
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
