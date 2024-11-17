from collections import OrderedDict
import torch.nn.functional as F
import normflows as nf
import torch
import torch.nn as nn
from normflows.flows import Planar, Radial
from torch.nn import Sequential
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, GCNConv
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout

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
            Shape (batch_size, CORE_NUM, r_dim)
        """
        r = self.aggregate(r)  # permutation-invariant
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return torch.distributions.Normal(mu, sigma)

class LatentEncoder(nn.Module):
    def __init__(self, num_classes, embed_size=100, two=True):
        super(LatentEncoder, self).__init__()
        self.nc = num_classes
        self.embed_size = embed_size
        if two:
            nsum = 3 * embed_size
        else:
            nsum = embed_size + self.nc

        self.rel_fc1 = nn.Linear(nsum, embed_size)


    def forward(self, inputs, y, two=True):
        size = inputs.shape   # B CORE_NUMS D
        x = inputs.contiguous() #.view(size[0], size[1])

        if two:
            label = y.unsqueeze(dim=1).repeat(1, self.embed_size)
        else:
            label = F.one_hot(y.to(torch.int64),num_classes=self.nc)

        x = torch.cat([x, label], dim=-1)
        x = self.rel_fc1(x)

        return x  # (B, CORE_NUMS, D+NC) -> B CORE_NUMS 100

class Flow(nn.Module):
    def __init__(self, latent_size, flow, K):
        super().__init__()
        if flow == "Planar":
            flows = [Planar((latent_size,)) for _ in range(K)]
        elif flow == "Radial":
            flows = [Radial((latent_size,)) for _ in range(K)]
        elif flow == "RealNVP":
            flows = []
            b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
            for i in range(K):
                s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
                t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
                if i % 2 == 0:
                    flows += [nf.flows.MaskedAffineFlow(b, t, s)]
                else:
                    flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        self.flows = nn.ModuleList(flows)

    def forward(self, z, base_dist, prior=None):
        ld = 0.0
        p_0 = torch.sum(base_dist.log_prob(z), -1)
        for flow in self.flows:
            z, ld_ = flow(z)
            ld += ld_
        # z = z.squeeze_()
        # KLD including logdet term
        if prior:
            kld = p_0 - torch.sum(prior.log_prob(z), -1) - ld.view(-1)
        else:
            kld = None

        return z, kld
