import torch
import numpy as np
import os
import random
import argparse
from exp import Exp
from logger import create_logger
from utils.config import load_config, save_config, list2str


seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='Spiking Graph Neural Networks on Riemannian Manifold')

# Experiment settings
parser.add_argument('--task', type=str, default='NC',
                    choices=['NC', 'LP'])
parser.add_argument('--dataset', type=str, default='Physics',
                    choices=['computers', 'photo', 'KarateClub', 'CS', 'Physics'])
parser.add_argument('--root_path', type=str, default='./datasets')
parser.add_argument('--eval_freq', type=int, default=10)
parser.add_argument('--exp_iters', type=int, default=5)
parser.add_argument('--log_path', type=str, default="./results/Physics.log")
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--w_decay', type=float, default=0.0)
parser.add_argument('--use_MS', action='store_false')

# Base Params
parser.add_argument('--use_product', action='store_true')
parser.add_argument('--manifold', type=str, nargs='+', default=['lorentz'],
                    help='Choose in combination [euclidean, lorentz, sphere]')
parser.add_argument('--neuron', type=str, default='IF', choices=['IF', 'LIF'], help="Which neuron to use")
parser.add_argument('--T', type=int, default=10, help="latency of neuron")
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--embed_dim', type=int, nargs='+', default=[32], help='embedding dimension')
parser.add_argument('--step_size', type=float, default=0.01, help='step size for tangent vector')
parser.add_argument('--v_threshold', type=float, default=1.0, help='threshold for neuron')
parser.add_argument('--delta', type=float, default=0.05, help='For LIF neuron')
parser.add_argument('--tau', type=float, default=2.)
parser.add_argument('--dropout', type=float, default=0.0)

# Node Classification
parser.add_argument('--lr_cls', type=float, default=0.01)
parser.add_argument('--w_decay_cls', type=float, default=0)
parser.add_argument('--epochs_cls', type=int, default=200)
parser.add_argument('--patience_cls', type=int, default=3)

# Link Prediction
parser.add_argument('--lr_lp', type=float, default=0.01)
parser.add_argument('--w_decay_lp', type=float, default=0)
parser.add_argument('--epochs_lp', type=int, default=200)
parser.add_argument('--patience_lp', type=int, default=3)
parser.add_argument('--t', type=float, default=1., help='for Fermi-Dirac decoder')
parser.add_argument('--r', type=float, default=2., help='Fermi-Dirac decoder')
# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')

configs = parser.parse_args()
results_dir = f"./results/logs"
log_path = f"{results_dir}/{configs.task}_{list2str(configs.manifold)}_{configs.dataset}.log"
configs.log_path = log_path
if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)
json_dir = f"./configs/{configs.task}/{configs.dataset}"
json_path = f"{json_dir}/{list2str(configs.manifold)}.json"
if not os.path.exists(json_dir):
    os.makedirs(json_dir, exist_ok=True)
times_dir = f"./results/times"
if not os.path.exists(times_dir):
    os.makedirs(times_dir, exist_ok=True)

# print(f"Saving config file: {json_path}")
# save_config(vars(configs), json_path)
if os.path.exists(json_path):
    print(f"Loading config file: {json_path}")
    configs = load_config(vars(configs), json_path)
else:
    print(f"Saving config file: {json_path}")
    save_config(vars(configs), json_path)

print(f"Log path: {configs.log_path}")
logger = create_logger(configs.log_path)
logger.info(configs)

exp = Exp(configs)
exp.train()
torch.cuda.empty_cache()