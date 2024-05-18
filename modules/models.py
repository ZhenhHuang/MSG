import torch
import torch.nn as nn
from modules.spike_encoder import GNNSpikeEncoder, PoissonSpikeEncoder
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from modules.layers import RiemannianSGNNLayer, RSEncoderLayer
from manifolds.lorentz import Lorentz
from manifolds.euclidean import Euclidean
from geoopt.tensor import ManifoldParameter
import math


class RiemannianSpikeGNN(nn.Module):
    def __init__(self, manifold, T, n_layers, step_size, in_dim, embed_dim,
                 n_classes, v_threshold=1.0, dropout=0.1, neuron="IF", delta=0.05, tau=2,
                 self_train=False, task='NC', use_MS=True, device=None):
        super(RiemannianSpikeGNN, self).__init__()
        if isinstance(manifold, Lorentz):
            embed_dim += 1
        self.manifold = manifold
        self.step_size = step_size
        self.self_train = self_train
        self.task = task
        self.encoder = RSEncoderLayer(manifold, T, in_dim, embed_dim, neuron=neuron, delta=delta, tau=tau,
                                      step_size=step_size, v_threshold=v_threshold,
                                      dropout=dropout, use_MS=use_MS)
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            self.layers.add_module(f"layer_{i}",
                                   RiemannianSGNNLayer(manifold, embed_dim, neuron=neuron, delta=delta, tau=tau,
                                                       step_size=step_size, v_threshold=v_threshold,
                                                       dropout=dropout, use_MS=use_MS)
                                   )
        # self.fc = nn.Linear(embed_dim, n_classes, bias=False) if task == "NC" else None
        self.fc = RiemannianClassifier(manifold, n_classes, embed_dim, device) if task == "NC" else None

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
            z = self.manifold.projx(z)

        if self.task == 'NC' and self.self_train is False:
            # z = self.manifold.proju0(self.manifold.logmap0(z))
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


class RiemannianClassifier(nn.Module):
    def __init__(self, manifold, num_classes, n_dim, device):
        super(RiemannianClassifier, self).__init__()
        self.manifold = manifold
        self.points = ManifoldParameter(self.manifold.random_normal((num_classes, n_dim), std=1./math.sqrt(n_dim),
                                                                    device=device, dtype=torch.get_default_dtype()),
                                        manifold, requires_grad=True)
        # self.w = nn.Linear(num_classes, num_classes, bias=False)

    def forward(self, x):
        return -self.manifold.dist(self.manifold.projx(self.points), x.unsqueeze(1))


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
