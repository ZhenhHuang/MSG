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
    def count_layer(m, x, y):
        # your rule here
        pass

    def count_encoder(m, x, y):
        # your rule here
        pass

    """
    macs, params = profile(method.model,
                        inputs=(method.cache.X, method.cache.A))
    """
    from modules.models import RiemannianSGNNLayer, RSEncoderLayer
    model.eval()
    flops, params = profile(model,
                            inputs=(data, ),
                            # custom_ops={RSEncoderLayer: count_layer
                            #     , RiemannianSGNNLayer: count_encoder}
                            )
    print("num of nodes: ", data["features"].shape[0])
    flops = flops / data["features"].shape[0]
    flops, params = clever_format([flops, params], "%.4f")
    return flops, params


class OutputExtractor(nn.Module):
    def __init__(self, index):
        super(OutputExtractor, self).__init__()
        self.index = index

    def forward(self, output: tuple):
        return output[self.index]