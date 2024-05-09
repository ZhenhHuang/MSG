import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import networkx as nx
import matplotlib.pyplot as plt
import random
from thop import profile
from thop import clever_format


def cal_accuracy(preds, trues):
    preds = torch.argmax(preds, dim=-1)
    correct = (preds == trues).sum()
    return correct / len(trues)


def cal_F1(preds, trues):
    preds = torch.argmax(preds, dim=-1)
    weighted_f1 = f1_score(trues, preds, average='weighted')
    macro_f1 = f1_score(trues, preds, average='macro')
    return weighted_f1, macro_f1


def cal_AUC_AP(scores, trues):
    auc = roc_auc_score(trues, scores)
    ap = average_precision_score(trues, scores)
    return auc, ap


def calc_params(model, data):
    def count_layer(m, input, output):
        # your rule here
        s, z, edge_index = input
        s_o, z_o = output
        m.total_params += z_o.shape[-1] * z.shape[-1]
        m.total_ops += s_o.sum().item() * 3.7

    def count_encoder(m, input, output):
        # your rule here
        x, edge_index = input
        s_o, z_o = output
        m.total_params += z_o.shape[-1] * x.shape[-1]
        m.total_ops += 4.6 * (z_o.shape[-1] * edge_index.shape[-1] +
                              2 * z_o.shape[0] * z_o.shape[-1] + z_o.shape[0] * z_o.shape[-1] * x.shape[-1] / 64)
        m.total_ops += 3.7 * s_o.sum().item()


    """
    macs, params = profile(method.model,
                        inputs=(method.cache.X, method.cache.A))
    """
    from modules.models import RiemannianSGNNLayer, RSEncoderLayer
    model.eval()
    energy, params = profile(model,
                            inputs=(data, ),
                            custom_ops={RSEncoderLayer: count_encoder
                                , RiemannianSGNNLayer: count_layer}
                            )
    params = clever_format([params], "%.4f")
    energy = f"{energy * 1e-9} mJ"
    return energy, params


class OutputExtractor(nn.Module):
    def __init__(self, index):
        super(OutputExtractor, self).__init__()
        self.index = index

    def forward(self, output: tuple):
        return output[self.index]