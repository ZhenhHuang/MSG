import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from geoopt.optim import RiemannianAdam
from manifolds import Lorentz, Sphere, Euclidean, ProductSpace
from modules.models import FermiDiracDecoder, RiemannianSpikeGNN
from spikingjelly.clock_driven.functional import reset_net
from utils.eval_utils import cal_accuracy, cal_F1, cal_AUC_AP, calc_params, OutputExtractor
from utils.data_utils import load_data, mask_edges
from utils.train_utils import EarlyStopping
from logger import create_logger
from utils.config import list2str
import time
import os


class Exp:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def send_device(self, data):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)

    def train(self):
        logger = create_logger(self.configs.log_path)
        device = self.device
        data = load_data(self.configs.root_path, self.configs.dataset)
        self.send_device(data)

        if self.configs.task == "NC":
            vals = []
            accs = []
            wf1s = []
            mf1s = []
        elif self.configs.task == "LP":
            aucs = []
            aps = []
        for exp_iter in range(self.configs.exp_iters):
            logger.info(f"\ntrain iters {exp_iter}")

            m_tuple = []
            for i, m_str in enumerate(self.configs.manifold):
                if m_str == "euclidean":
                    m_tuple.append((Euclidean(), self.configs.embed_dim[i]))
                elif m_str == 'lorentz':
                    m_tuple.append((Lorentz(), self.configs.embed_dim[i]))
                elif m_str == 'sphere':
                    m_tuple.append((Sphere(), self.configs.embed_dim[i]))

            if self.configs.use_product:
                print('using product space')
                manifold = ProductSpace(*m_tuple)
            else:
                manifold = m_tuple[0][0]

            model = RiemannianSpikeGNN(manifold, T=self.configs.T, n_layers=self.configs.n_layers,
                                       in_dim=data["num_features"], neuron=self.configs.neuron,
                                       embed_dim=sum(self.configs.embed_dim), n_classes=data["num_classes"],
                                       step_size=self.configs.step_size, v_threshold=self.configs.v_threshold,
                                       dropout=self.configs.dropout, self_train=self.configs.self_train,
                                       task=self.configs.task, use_MS=self.configs.use_MS, tau=self.configs.tau).to(
                device)

            logger.info("--------------------------Training Start-------------------------")
            if self.configs.self_train:
                energy, params = calc_params(model, data)
                logger.info(f"energy: {energy}, params: {params}")
                model = self.pre_train(data, model, logger)
            if self.configs.task == 'NC':
                energy, params = calc_params(model, data)
                statistic_str = f"Energy results for {self.configs}: \n energy: {energy}, params: {params} \n"
                logger.info(statistic_str)
                with open("./results/energy.txt", "a") as f:
                    f.write(statistic_str)
                f.close()
                best_val, test_acc, test_weighted_f1, test_macro_f1 = self.train_cls(data, model, logger, exp_iter)
                energy, params = calc_params(model, data)
                logger.info(f"energy: {energy}, params: {params}")
                logger.info(
                    f"val_accuracy={best_val.item() * 100: .2f}%, test_accuracy={test_acc.item() * 100: .2f}%")
                logger.info(
                    f"\t\t weighted_f1={test_weighted_f1 * 100: .2f}%, macro_f1={test_macro_f1 * 100: .2f}%")
                vals.append(best_val.item())
                accs.append(test_acc.item())
                wf1s.append(test_weighted_f1)
                mf1s.append(test_macro_f1)
            elif self.configs.task == 'LP':
                # energy, params = calc_params(model, data)
                # logger.info(f"energy: {energy}, params: {params}")
                _, test_auc, test_ap = self.train_lp(data, model, logger, exp_iter)
                # energy, params = calc_params(model, data)
                # logger.info(f"energy: {energy}, params: {params}")
                logger.info(
                    f"test_auc={test_auc * 100: .2f}%, test_ap={test_ap * 100: .2f}%")
                aucs.append(test_auc)
                aps.append(test_ap)
            else:
                raise NotImplementedError

        if self.configs.task == "NC":
            result_str = f"NC results for {self.configs}: \n" + \
                         f"valid results: {np.mean(vals) * 100: .2f}~{np.std(vals) * 100: .2f} \n" + \
                         f"best test ACC: {np.max(accs) * 100: .2f}\n" + \
                         f"test results: {np.mean(accs) * 100: .2f}~{np.std(accs) * 100: .2f}\n" + \
                         f"test weighted-f1: {np.mean(wf1s) * 100: .2f}~{np.std(wf1s) * 100: .2f}\n" + \
                         f"test macro-f1: {np.mean(mf1s) * 100: .2f}~{np.std(mf1s) * 100: .2f}\n"
            logger.info(result_str)
            with open("./results/NC_results.txt", "a") as f:
                f.write(result_str)
            f.close()
        elif self.configs.task == "LP" or self.configs.task == 'Motif':
            result_str = f"LP results for {self.configs}: \n" + \
                         f"test AUC: {np.mean(aucs) * 100: .2f}~{np.std(aucs)* 100: .2f} \n" + \
                         f"test AP: {np.mean(aps) * 100: .2f}~{np.std(aps) * 100: .2f} \n"
            logger.info(result_str)
            with open("./results/LP_results.txt", "a") as f:
                f.write(result_str)
            f.close()

    def pre_train(self, data, model, logger):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.configs.lr,
                                     weight_decay=self.configs.w_decay)
        all_times = []
        all_backward_times = []
        for epoch in range(1, self.configs.epochs + 1):
            model.train()
            optimizer.zero_grad()
            epoch_time = time.time()
            _, loss = model(data)
            backward_time = time.time()
            loss.backward()
            backward_time = time.time() - backward_time
            optimizer.step()
            epoch_time = time.time() - epoch_time
            logger.info(f"Epoch {epoch}: train_loss={loss.item()}"
                        f", \n epoch_time={epoch_time} s, backward_time={backward_time} s")
            all_times.append(epoch_time)
            all_backward_times.append(backward_time)
        model.self_train = False
        return model

    def cal_cls_loss(self, model, data, mask):
        out = model(data)[mask]
        # correct = out.gather(1, data['labels'][mask].unsqueeze(-1))
        loss = F.cross_entropy(out, data["labels"][mask])
        # target = torch.ones_like(correct)
        # loss = F.margin_ranking_loss(correct, out, target, margin=self.configs.margin, reduction="mean")
        acc = cal_accuracy(out, data["labels"][mask])
        weighted_f1, macro_f1 = cal_F1(out.detach().cpu(), data["labels"][mask].detach().cpu())
        return loss, acc, weighted_f1, macro_f1

    def train_cls(self, data, model_cls, logger, exp_iter: int):
        best_acc = 0.
        all_times = []
        all_backward_times = []
        early_stop = EarlyStopping(self.configs.patience_cls, verbose=True)
        optimizer = RiemannianAdam(model_cls.parameters(), lr=self.configs.lr_cls,
                                   weight_decay=self.configs.w_decay_cls)
        for epoch in range(1, self.configs.epochs_cls + 1):
            model_cls.train()
            optimizer.zero_grad()
            epoch_time = time.time()
            loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, data, data["train_mask"])
            backward_time = time.time()
            loss.backward()
            backward_time = time.time() - backward_time
            optimizer.step()
            epoch_time = time.time() - epoch_time
            logger.info(f"Epoch {epoch}: train_loss={loss.item()}, train_accuracy={acc}"
                        f", \n epoch_time={epoch_time} s, backward_time={backward_time} s")
            all_times.append(epoch_time)
            all_backward_times.append(backward_time)
            if self.configs.use_MS is not True:
                reset_net(model_cls)

            if epoch % self.configs.eval_freq == 0:
                model_cls.eval()
                val_loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, data, data['val_mask'])
                logger.info(f"Epoch {epoch}: val_loss={val_loss.item()}, val_accuracy={acc}")
                if acc > best_acc:
                    best_acc = acc
                early_stop(val_loss, model_cls, save=False, path=
                f"{self.configs.task}_{self.configs.dataset}_{list2str(self.configs.manifold)}_{exp_iter}.pt")
                if early_stop.early_stop:
                    break

        avg_train_time = np.mean(all_times)
        avg_backward_time = np.mean(all_backward_times)
        np.save(file=f"./results/times/backward_time_"
                     f"{self.configs.dataset}_{self.configs.use_MS}_{list2str(self.configs.manifold)}_{self.configs.T}.npy",
                arr=np.array(all_backward_times))
        time_str = f"Average Time: {avg_train_time} s/epoch, Average Backward Time: {avg_backward_time} s/epoch"
        logger.info(time_str)
        time_str = f"{self.configs.task}_{self.configs.dataset}_{time_str}\n"
        with open('time.txt', 'a') as f:
            f.write(time_str)
        f.close()
        test_loss, test_acc, test_weighted_f1, test_macro_f1 = self.cal_cls_loss(model_cls, data, data['test_mask'])
        return best_acc, test_acc, test_weighted_f1, test_macro_f1

    def cal_lp_loss(self, embeddings, model, pos_edges, neg_edges):
        if not isinstance(model.manifold, Lorentz):
            pos_dists = model.manifold.dist(embeddings[pos_edges[0]], embeddings[pos_edges[1]])
            pos_scores = torch.sigmoid((self.configs.r - pos_dists) / self.configs.t)
            neg_dists = model.manifold.dist(embeddings[neg_edges[0]], embeddings[neg_edges[1]])
            neg_scores = torch.sigmoid((self.configs.r - neg_dists) / self.configs.t)
            loss = F.binary_cross_entropy(pos_scores.clip(0.01, 0.99), torch.ones_like(pos_scores)) + \
                   F.binary_cross_entropy(neg_scores.clip(0.01, 0.99), torch.zeros_like(neg_scores))
        else:
            pos_scores = model.manifold.inner(None, embeddings[pos_edges[0]], embeddings[pos_edges[1]])
            neg_scores = model.manifold.inner(None, embeddings[neg_edges[0]], embeddings[neg_edges[1]])
            loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
                   F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        label = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.detach().cpu().numpy()) + list(neg_scores.detach().cpu().numpy())
        auc, ap = cal_AUC_AP(preds, label)
        return loss, auc, ap

    def train_lp(self, data, model, logger, exp_iter: int):
        optimizer_lp = torch.optim.Adam(model.parameters(), lr=self.configs.lr_lp,
                                        weight_decay=self.configs.w_decay_lp)
        # decoder = FermiDiracDecoder(self.configs.r, self.configs.t).to(self.device)
        best_ap = 0
        early_stop = EarlyStopping(self.configs.patience_cls)
        time_before_train = time.time()
        for epoch in range(1, self.configs.epochs_lp + 1):
            t = time.time()
            model.train()
            optimizer_lp.zero_grad()
            embeddings = model(data)
            neg_edge_train = data["neg_edges_train"][:,
                             np.random.randint(0, data["neg_edges_train"].shape[1], data["pos_edges_train"].shape[1])]
            loss, auc, ap = self.cal_lp_loss(embeddings, model, data["pos_edges_train"], neg_edge_train)
            loss.backward()
            optimizer_lp.step()
            logger.info(
                f"Epoch {epoch}: train_loss={loss.item()}, train_AUC={auc}, train_AP={ap}, time={time.time() - t}")
            if self.configs.use_MS is not True:
                reset_net(model)
            if epoch % self.configs.eval_freq == 0:
                model.eval()
                val_loss, auc, ap = self.cal_lp_loss(embeddings, model, data["pos_edges_val"], data["neg_edges_val"])
                logger.info(f"Epoch {epoch}: val_loss={val_loss.item()}, val_AUC={auc}, val_AP={ap}")
                if ap > best_ap:
                    best_ap = ap
                early_stop(val_loss, model, save=False, path=
                f"{self.configs.task}_{self.configs.dataset}_{list2str(self.configs.manifold)}_{exp_iter}.pt")
                if early_stop.early_stop:
                    break
        avg_train_time = (time.time() - time_before_train) / epoch
        time_str = f"Average Time: {avg_train_time} s/epoch"
        logger.info(time_str)
        time_str = f"{self.configs.task}_{self.configs.dataset}_{time_str}\n"
        with open('time.txt', 'a') as f:
            f.write(time_str)
        f.close()
        test_loss, test_auc, test_ap = self.cal_lp_loss(embeddings, model, data["pos_edges_test"],
                                                        data["neg_edges_test"])
        return test_loss, test_auc, test_ap
