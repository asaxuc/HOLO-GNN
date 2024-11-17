import torch
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, GCNConv
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.nn import Linear
import torch.nn.functional as F
from torch_scatter import segment_csr
import torch.nn as nn


class GIN(torch.nn.Module):
    def __init__(self, args, ins, outs, use_drop=False):
        super(GIN, self).__init__()

        self.conv1 = GINConv(
            Sequential(Linear(ins, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))

        self.conv4 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))

        self.conv5 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, outs), ReLU()))

        self.dropout = Dropout(p=0.2)
        self.use_drop = use_drop

    def forward(self, x, adj_t, batch):
        x = self.conv1(x, adj_t)
        if self.use_drop:
            x = self.dropout(x)
        x = self.conv4(x, adj_t)
        x = self.conv5(x, adj_t)

        x = segment_csr(x, batch, reduce="sum")

        return x

class GIN_WOP(torch.nn.Module):
    def __init__(self, args, zdim=None, use_drop=False):
        super(GIN_WOP, self).__init__()

        zzl = zdim if zdim else args.n_hidden

        self.conv1 = GINConv(
            Sequential(Linear(args.n_feat, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))

        self.conv4 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))

        self.conv5 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, zzl), ReLU()))

        self.dropout = Dropout(p=0.2)
        self.use_drop = use_drop

    def forward(self, x, adj_t, batch, sub_mask=None):
        x = self.conv1(x, adj_t)
        if self.use_drop:
            x = self.dropout(x)
        x = self.conv2(x, adj_t)
        x = self.conv3(x, adj_t)
        x = self.conv4(x, adj_t)
        x = self.conv5(x, adj_t)

        if sub_mask is not None:
            t = torch.zeros_like(x).cuda()
            t[sub_mask] += x[sub_mask]
            x = t

        xr = segment_csr(x, batch, reduce="sum")
        return xr, x

class MLP_Classifier(torch.nn.Module):
    def __init__(self, args):
        super(MLP_Classifier, self).__init__()
        self.lin1 = Linear(args.n_hidden, args.n_hidden)
        self.lin2 = Linear(args.n_hidden, args.n_class)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)


class Vanilla_GCN(nn.Module):
    def __init__(self, in_channels, channels, dropout, act, improved=True):
        super().__init__()
        self.act = act
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, channels, improved)
        self.conv2 = GCNConv(channels, channels, improved)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
