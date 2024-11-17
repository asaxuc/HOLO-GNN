from collections import OrderedDict
from func.flow import Flow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence

def get_hyperdata(indices_matrix):
    indice_matrix = indices_matrix.float()
    W_e_diag = torch.ones(indice_matrix.size()[1] ,dtype=torch.float).cuda()  #

    D_e_diag = torch.sum(indice_matrix, 0)
    D_e_diag = D_e_diag.view((D_e_diag.size()[0]))

    D_v_diag = indice_matrix.mm(W_e_diag.view((W_e_diag.size()[0]), 1))
    D_v_diag = D_v_diag.view((D_v_diag.size()[0]))

    Theta = torch.diag(torch.pow(D_v_diag, -0.5)) @ \
            indice_matrix @ torch.diag(W_e_diag) @ \
            torch.diag(torch.pow(D_e_diag, -1)) @ \
            torch.transpose(indice_matrix, 0, 1) @ \
            torch.diag(torch.pow(D_v_diag, -0.5))  # indi 好说  实话说 We_diag就是 一维1，diag以下变 对角1

    Theta_inverse = torch.pow(Theta, -1)
    Theta_inverse[Theta_inverse == float("Inf")] = 0

    Theta_I = torch.diag(torch.pow(D_v_diag, -0.5)) @ \
              indice_matrix @ torch.diag(W_e_diag + torch.ones_like(W_e_diag)) @ \
              torch.diag(torch.pow(D_e_diag, -1)) @ \
              torch.transpose(indice_matrix, 0, 1) @ \
              torch.diag(torch.pow(D_v_diag, -0.5))

    Theta_I[Theta_I != Theta_I] = 0
    Theta_I_inverse = torch.pow(Theta_I, -1)
    Theta_I_inverse[Theta_I_inverse == float("Inf")] = 0

    Laplacian = torch.eye(Theta.size()[0]).cuda() - Theta

    fourier_e, fourier_v = torch.symeig(Laplacian, eigenvectors=True)
    # fourier_e, fourier_v = np.linalg.eig(Laplacian)

    wavelets = fourier_v @ torch.diag(torch.exp(-1.0 * fourier_e * 1)) @ torch.transpose(fourier_v, 0, 1)
    wavelets_inv = fourier_v @ torch.diag(torch.exp(fourier_e * 1)) @ torch.transpose(fourier_v, 0, 1)
    wavelets_t = torch.transpose(wavelets, 0, 1)
    # 根据那篇论文的评审意见，这里用wavelets_t或许要比wavelets_inv效果更好？

    wavelets[wavelets < 0.00001] = 0
    wavelets_inv[wavelets_inv < 0.00001] = 0
    wavelets_t[wavelets_t < 0.00001] = 0

    hypergraph = {"indice_matrix": indice_matrix,
                  "D_v_diag": D_v_diag,
                  "D_e_diag": D_e_diag,
                  "W_e_diag": W_e_diag,  # hyperedge_weight_flat
                  "laplacian": Laplacian,
                  "fourier_v": fourier_v,
                  "fourier_e": fourier_e,
                  "wavelets": wavelets,
                  "wavelets_inv": wavelets_inv,
                  "wavelets_t": wavelets_t,
                  "Theta": Theta,
                  "Theta_inv": Theta_inverse,
                  "Theta_I": Theta_I,
                  "Theta_I_inv": Theta_I_inverse,
                  }
    return hypergraph

class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """

    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def aggregate(self, r):
        return torch.mean(r, dim=0)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, few, r_dim)
        """
        r = self.aggregate(r)  # permutation-invariant
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return torch.distributions.Normal(mu, sigma)



class LatentEncoder(nn.Module):
    def __init__(self,  embed_size=100, num_hidden1=500, num_hidden2=200, r_dim=100, dropout_p=0.5):
        super(LatentEncoder, self).__init__()
        self.embed_size = embed_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(2 * embed_size + 1, num_hidden1)),
            # ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden1, num_hidden2)),
            # ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, r_dim)),
            # ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, bs, ns, inputs, y):
        size = inputs.shape
        x = inputs.contiguous()  # (B, dim * 2)

        label = F.one_hot(y,num_classes=ns).unsqueeze(dim=0).repeat(x.shape[0] // bs)
        x = torch.cat([x, label], dim=-1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)

        return x.view(bs, -1, x.shape[-1])  # (B, guding, r_dim)


# class BWM(nn.Module):
#     def __init__(self, hyper_ratio, topk, num_classes, dim):
#         super(BWM,self).__init__()
#         self.hyper_ratio = hyper_ratio
#         self.topk = topk
#         self.hyper_map = nn.Parameter(torch.randn(num_classes, dim))
#
#
#
#     def forward(self, sbatch, y_query):
#         hyper_clss = self.hyper_map[y_query] # b D
#         hyper_clss = hyper_clss.unsqueeze(dim=1) # b B D
#         x = sbatch["x"].unsqueeze(dim=0) # B D
#         edge_index= sbatch["edge_index"]
#         edge_weight = sbatch["edge_weight"]
#         batch = sbatch["batch"]
#         sim =  (x * hyper_clss).sum(dim=-2) # b B
#
#         thrices = sim.topk(self.topk,dim=1)[0].unsqueeze(dim=1)
#         hyper_matrix = (sim > thrices).to(torch.int).T
#         hypergraph = get_hyperdata(hyper_matrix)
#         hyper_x =
#         return hypergraph



class IWM(nn.Module):
    def __init__(self, hyper_ratio, hyper_per_num, topk, dim, num_classes, bs):
        super(IWM,self).__init__()
        self.hyper_ratio = hyper_ratio
        self.hyper_per_num = hyper_per_num
        self.topk = topk
        self.bs = bs
        self.nc = num_classes
        self.encoder = LatentEncoder(dim)
        self.flows = Flow(self.z_dim, 'Planar', 10)
        self.xy_to_musigma = MuSigmaEncoder(100,100)
        assert self.hyper_per_num < self.topk


    def forward(self, s_batch, y_batch, s_query, y_query, istest=False):
        x = s_batch["x"]
        xq = s_query["x"]

        # in dim
        if istest:
            r = self.encoder(self.bs, self.nc, x, y_batch)
            target_dist = self.xy_to_musigma(r)
            z = target_dist.sample()
            if self.np_flow != 'none':
                z, _ = self.flows(z, target_dist)
        else:
            r = self.encoder(self.bs, self.nc, x, y_batch)
            rq = self.encoder(self.bs, self.nc, xq, y_query)
            context_dist = self.xy_to_musigma(r)
            target_dist = self.xy_to_musigma(torch.cat((r, rq),dim=0))
            z = target_dist.rsample()
            if self.np_flow != 'none':
                z, kld = self.flows(z, target_dist, context_dist)
            else:
                kld = kl_divergence(target_dist, context_dist).sum(-1)

            return z, kld








