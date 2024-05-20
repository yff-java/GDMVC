import torch
import numpy as np
import utils
from torch import nn


class AdaGAE(torch.nn.Module):
    def __init__(self, layer_dims, z_pass_linear=False):
        super().__init__()
        self.w1 = self.get_weight_initial([layer_dims[0], layer_dims[1]])
        self.w2 = self.get_weight_initial([layer_dims[1], layer_dims[2]])

        if z_pass_linear:
            self.z_pass_linear1 = torch.nn.Linear(layer_dims[2], layer_dims[1])
            self.z_pass_linear2 = torch.nn.Linear(layer_dims[1], layer_dims[2])

    def get_weight_initial(self, shape):
        bound = np.sqrt(6.0 / (shape[0] + shape[1]))
        ini = torch.rand(shape) * 2 * bound - bound
        return torch.nn.Parameter(ini, requires_grad=True)

    def forward(self, xi, Laplacian):
        # 编码
        embedding = Laplacian.mm(xi.matmul(self.w1))
        embedding = torch.nn.functional.relu(embedding)
        embedding = Laplacian.mm(embedding.matmul(self.w2))

        # 重构
        distances = utils.distance(embedding.t(), embedding.t())
        softmax = torch.nn.Softmax(dim=1)
        recons_w = softmax(-distances)
        return embedding, recons_w + 10**-10

    def cal_loss(self, raw_weights, recons, weights, embeding, lam):
        re_loss = raw_weights * torch.log(raw_weights / recons + 10**-10)
        re_loss = re_loss.sum(dim=1)
        re_loss = re_loss.mean()

        size = embeding.shape[0]
        degree = weights.sum(dim=1)
        L = torch.diag(degree) - weights
        tr_loss = torch.trace(embeding.t().matmul(L).matmul(embeding)) / size
        return re_loss, tr_loss


class AdaGAEMV(torch.nn.Module):
    def __init__(self, X, layers, device):
        layers_list = [[x.shape[1]] + layers for x in X]
        self.gae_list = [AdaGAE(layer).to(device) for layer in layers_list]

    def forward(self, X, lapacian_mv):
        embedding_list = []
        recons_w_list = []
        for i in range(len(X)):
            embedding, recons_w = self.gae_list[i](X[i], lapacian_mv[i])
            embedding_list.append(embedding)
            recons_w_list.append(recons_w)
        return embedding_list, recons_w_list

    def cal_loss(self, raw_weights_mv, recons_w_list, weights_mv, embedding_list, lam):
        re_loss_list = []
        tr_loss_list = []
        for i in range(len(recons_w_list)):
            re_loss, tr_loss = self.gae_list[i].cal_loss(
                raw_weights_mv[i],
                recons_w_list[i],
                weights_mv[i],
                embedding_list[i],
                lam,
            )
            re_loss_list.append(re_loss)
            tr_loss_list.append(tr_loss)
        re_loss = sum(re_loss_list) / len(re_loss_list)
        tr_loss = sum(tr_loss_list) / len(tr_loss_list)
        return re_loss, tr_loss


class AutoEncoder(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[2]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(layers[2], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[0]),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_re = self.decoder(z)
        return z, x_re


class AutoEncoderMV(torch.nn.Module):
    def __init__(self, X, layers, device):
        layers_list = [[x.shape[1]] + layers for x in X]
        self.ae_list = [AutoEncoder(layer).to(device) for layer in layers_list]

    def forward(self, X):
        z_list = []
        x_re_list = []
        for i in range(len(X)):
            z, x_re = self.ae_list[i](X[i])
            z_list.append(z)
            x_re_list.append(x_re)
        return z_list, x_re_list

    def cal_re_loss(self, X, x_re_list):
        re_loss_list = []
        for i in range(len(X)):
            re_loss = torch.sum((X[i] - x_re_list[i]) ** 2, dim=1)
            re_loss = re_loss.mean()
            re_loss_list.append(re_loss)
        re_loss = sum(re_loss_list) / len(re_loss_list)
        return re_loss
