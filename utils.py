import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import networkx as nx
import matplotlib.pyplot as plt
import random


EPS = {torch.float32: 1e-4, torch.float64: 1e-7}

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


class SinhDiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y_stable = torch.ones_like(x)
        y = torch.sinh(x) / x
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (x * torch.cosh(x) - torch.sinh(x)) / x ** 2
        y_stable = torch.zeros_like(x)
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y) * grad_output


sinh_div = SinhDiv.apply