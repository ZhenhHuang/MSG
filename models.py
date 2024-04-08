import torch
import torch.nn as nn
from spike_encoder import GNNSpikeEncoder, PoissonSpikeEncoder
from spikingjelly.clock_driven.neuron import MultiStepLIFNode


class SpikeClassifier(nn.Module):
    def __init__(self, T, backbone, n_layers, n_neurons, hidden_neurons,
                 embed_neurons, num_classes, n_heads, dropout):
        super(SpikeClassifier, self).__init__()
        self.encoder = GNNSpikeEncoder(T, backbone, n_layers, n_neurons, hidden_neurons, embed_neurons, n_heads,
                                       dropout)
        self.fc = nn.Linear(out_neurons, num_classes)
        self.lif = MultiStepLIFNode(detach_reset=True)
        self.T = T

    def forward(self, data):
        x, edge_index = data['features'], data['edge_index']
        x = self.encoder(x, edge_index)  # [T, N, D]
        x = self.lif(self.fc(x))    # [T, N, C]
        return torch.log(x.sum(0) / self.T + 1e-6)


class SpikeLinkPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        pass