import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven.encoding import PoissonEncoder
from gnn_backbone import choose_backbone


class GNNSpikeEncoder(nn.Module):
    def __init__(self, T, backbone, n_layers, n_neurons, hidden_neurons,
                 out_neurons, n_heads, dropout):
        super(GNNSpikeEncoder, self).__init__()
        self.gnn = choose_backbone(backbone, n_layers, n_neurons, hidden_neurons, out_neurons,
                                   n_heads=n_heads, drop_edge=dropout, drop_node=dropout)
        self.lif = MultiStepLIFNode(detach_reset=True)
        self.T = T

    def forward(self, x, edge_index):
        t_output = []
        for t in range(self.T):
            xt = self.gnn(x, edge_index)
            t_output.append(xt)
        xt = torch.stack(t_output, dim=0)
        spike_output = self.lif(xt)
        return spike_output


class PoissonSpikeEncoder(nn.Module):
    def __init__(self, T, backbone, n_layers, n_neurons, hidden_neurons,
                 out_neurons, n_heads, dropout):
        super(PoissonSpikeEncoder, self).__init__()
        self.gnn = choose_backbone(backbone, n_layers, n_neurons, hidden_neurons, out_neurons,
                                   n_heads=n_heads, drop_edge=dropout, drop_node=dropout)
        self.encoder = PoissonEncoder()
        self.T = T

    def forward(self, x, edge_index):
        x = self.gnn(x, edge_index)
        t_output = []
        for t in range(self.T):
            xt = self.encoder(x)
            t_output.append(xt)
        spike_output = torch.stack(t_output, dim=0)
        return spike_output


if __name__ == '__main__':
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='./data', name='Cora')
    # encoder = GNNSpikeEncoder(T=100, backbone='gcn', n_layers=1, n_neurons=dataset.num_features,
    #                           hidden_neurons=128, out_neurons=dataset.num_classes,
    #                           n_heads=1, dropout=0.5)
    encoder = PoissonSpikeEncoder(T=100, backbone='gcn', n_layers=1, n_neurons=dataset.num_features,
                              hidden_neurons=128, out_neurons=dataset.num_classes,
                              n_heads=1, dropout=0.5)
    spike_sequence = encoder(dataset.data.x, dataset.data.edge_index)