import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from spikingjelly.clock_driven import encoding


# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)

        return x


def GCNEncoder(dataset, T):
    data = dataset[0]

    # 定义GCN模型参数
    input_dim = dataset.num_node_features
    hidden_dim = 16
    # output_dim = dataset.num_classes
    output_dim = 32

    # 初始化模型
    model = GCN(input_dim, hidden_dim, output_dim)
    # 将模型切换到训练模式
    # model.train()

    # 获取模型的输出
    output = model(data)
    print(output.shape)

    encoder = encoding.PoissonEncoder()
    spike_seqs = []
    for t in range(T):
        spike_seqs.append(encoder(output).float())
    spike_seqs = torch.stack(spike_seqs, dim=0)
    return spike_seqs


if __name__ == '__main__':
    # 下载Cora数据集
    dataset = Planetoid(root='./data', name='Cora')
    print(dataset[0].num_nodes)
    tau = 100
    spike_seqs = GCNEncoder(dataset, tau)
    print(spike_seqs)
    print(spike_seqs.size())
