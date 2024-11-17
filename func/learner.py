import pickle
import random
from collections import OrderedDict

import numpy
from torch_geometric.data import Data

from func.HGNN_utils import generate_G_from_H
from torch_geometric.nn import TopKPooling
from func.HWNN import HWNN
import normflows as nf
from torch_geometric.utils import k_hop_subgraph, degree
from func.HSIC import hsic_regular
from einops import rearrange
from normflows.flows import Planar, Radial
from func.modulator import get_hyperdata
from models import *
import torch
from func.HGNN import HGNN, Shift_Intra
import torch.nn.functional as F
import torch_sparse
from func.flow import MuSigmaEncoder, LatentEncoder
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.utils import to_dense_adj, to_dense_batch, add_remaining_self_loops, add_self_loops, remove_self_loops, sort_edge_index, softmax ,subgraph
from torch.distributions import kl_divergence


class Flow(nn.Module):
    def __init__(self, bs, latent_size, flow, K):
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
        self.bs = bs

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


def smoothed(log_prob, targets, y, num_classes, epsilon=0.1, T=20):
    # N = targets.size(0)
    # smoothed_labels = torch.full(size=(N, num_classes),
    #                              fill_value=epsilon / (num_classes - 1)).cuda()
    # smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1),
    #                          value=1-epsilon)
    # log_prob = F.log_softmax(outputs / T, dim=1)
    N = targets.shape[0]
    adds = F.one_hot(y, num_classes=num_classes).cuda()
    smoothed_labels = F.one_hot(targets,num_classes=num_classes) * (1-epsilon) + adds * epsilon
    loss = - torch.sum(log_prob * smoothed_labels/ T) / N
    return loss



class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''

        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0

        self.nodegroup = 0

        self.K = 0
        self.sample_list = []
        self.unsample_list = []


class HyperConstru(nn.Module):
    def __init__(self, num_classes, num_hyper_per, topk_ratio, zdim):
        super(HyperConstru, self).__init__()
        self.topk_ratio = topk_ratio
        self.num_per_hyper = num_hyper_per
        self.hyper_params = nn.Parameter(torch.randn(num_classes, num_hyper_per, zdim))

    def forward(self, x, y): # y:选择的增强类，可能有多个. 我们首先按照1个来计算。1个！！
        hyper_clss = self.hyper_params[y, :, :].view(-1,self.hyper_params.shape[-1]) # enh*nh D
        # sim =  (x.unsqueeze(dim=0) * hyper_clss).sum(dim=-1) # B K D

        simi = rearrange(torch.einsum("bkd,ed->bke", x, hyper_clss),"B K E-> (B K) E").T

        thrices = torch.topk(simi, k=int(self.topk_ratio * simi.shape[-1]), dim=-1)[0][:,-1].unsqueeze(dim=1)
        hyper_matrix = (simi > thrices).to(torch.int).squeeze().float().T  # 需要aug的类，每次HWNN搞几个边，节点们！
        hyper_matrix = generate_G_from_H(hyper_matrix)
        # hypergraph = get_hyperdata(hyper_matrix)
        hyper_x = x.view(-1,x.shape[-1])

        return hyper_matrix, hyper_x


def construct_hyper(hyper_edge_query, x, num_hyper_edges=5): # hyper_edge_query C D
    # hyper zhong  keneng cunzai 0
    simi = rearrange(torch.einsum("bkd,ed->bke", x, hyper_edge_query), "B K E-> (B K) E").flatten() # assert C

    assert x.shape[-1] % num_hyper_edges == 0
    sorted_value, indices = torch.sort(simi,dim=0)
    hyper_matrix = torch.zeros((num_hyper_edges, simi.shape[0])).cuda()
    indices = indices.view(num_hyper_edges,-1).to(torch.long)

    hyper_matrix[torch.arange(num_hyper_edges).cuda()[:,None], indices] = 1
    hyper_matrix = generate_G_from_H(hyper_matrix.T)

    hyper_x = x.view(-1, x.shape[-1])
    return hyper_matrix, hyper_x

class Learner(nn.Module):
    def __init__(self, args, core_nums, final_core_num,  query_nums, act=F.relu): # (40,20,0...)
        super().__init__()

        rdim = zdim = args.n_hidden
        topk = 10
        num_per_hyper = 3
        hyper_ratio = 0.1

        self.args = args
        self.act = act
        self.core_nums = core_nums
        self.final_core_num = final_core_num
        self.topk = core_nums # 实际，10
        self.num_classes = args.n_class
        self.rdim = self.zdim = args.n_hidden
        self.K = 20

        self.down_param_k = nn.Parameter(torch.randn(20, args.n_hidden))
        self.feature_cali = nn.Linear(args.n_hidden, args.n_hidden)
        self.recon_cali = nn.Linear(args.n_hidden, args.n_hidden)
        self.conv = nn.Conv1d(args.n_hidden, args.n_hidden,3,1,1)

        self.query_nums = query_nums

        self.self_attn = torch.nn.MultiheadAttention(args.n_hidden, 4, dropout=0.2, batch_first=True)

        self.xy_to_mu_sigma = MuSigmaEncoder(rdim, zdim)  # xy_to_mu_sigma # MuSigmaEncoder(self.rdim, self.zdim)
        self.latent_encoder = LatentEncoder(args.n_class, embed_size=zdim) # latent_encoder # LatentEncoder(args.n_class, embed_size=args.n_hidden)
        self.hwnn =  HGNN(zdim) # HWNN(topk * args.batch_size, num_per_hyper, args.n_hidden ) # HWNN(self.topk * args.batch_size, args.n_hidden)
        self.gnn_down_1 = GIN_WOP(args, zdim)
        self.post_gnn = GIN(args, zdim, zdim)
        self.pooling = TopKPooling(zdim, 0.5)
        self.moto = self.query_nums // 2
        self.shift = Shift_Intra(zdim,zdim)

    def get_k_top_subgraphs(self, adj, batch):
        edge_index = torch.stack(adj.coo()[:2])

        samples = []
        for item in torch.unique(batch):
            plc = numpy.where(batch.detach().cpu().numpy()==item.detach().cpu().numpy())[0].tolist()
            sample = random.sample(plc, 1)
            samples += sample

        subset_mask, edge_index =  k_hop_subgraph(node_idx=samples, num_hops=self.args.hops, edge_index=edge_index)[:2]  # idx, edge_index

        return subset_mask, edge_index


    def forward(self, batch, encoder, classifier, switch):
        data, train_idx = batch['data'], batch['train_idx']
        if self.args.dataset != "REDDIT-BINARY":
            dx = data.x.cuda()
        else:
            dx = torch.ones((data.num_nodes, 1)).cuda()
        dadj_t = data.adj_t.cuda()
        dptr = data.ptr.cuda()
        dbatch = data.batch.cuda()
        dy = data.y.cuda()

        if switch == "ours":
            add_ones = ~(dy == self.args.heads[0])
            unadd_ones = (dy == self.args.heads[0])

            sub_mask, sub_edge_index = self.get_k_top_subgraphs(dadj_t, dbatch)

            x_down_1, x_down_wop = self.gnn_down_1(dx, dadj_t, dptr)
            feats = self.self_attn(x_down_1, self.down_param_k, self.down_param_k, need_weights=False)[0] # B D

            x_sub_down_1, x_sub_down_wop = self.gnn_down_1(dx, sub_edge_index, dptr, sub_mask)
            sub_feats = self.self_attn(x_sub_down_1, self.down_param_k, self.down_param_k, need_weights=False)[0] # B D

            f = [ix for ix in range(feats.shape[0])]
            random.shuffle(f)

            # # v3: separate indexing
            # fgot_all_x = torch.cat((sub_feats,feats), dim=-1)
            # fgot_all_x_neg = torch.cat((sub_feats, feats[f]), dim=-1)
            #
            # # tail
            # x_context_tail = fgot_all_x[add_ones]
            # x_context_tail = x_context_tail[:max(x_context_tail.shape[0]//2, 1)]
            # x_neg_context_tail = fgot_all_x_neg[add_ones]
            # x_neg_context_tail = x_neg_context_tail[:max(x_neg_context_tail.shape[0]//2, 1)]
            # y_context_tail = torch.ones((x_context_tail.shape[0])).cuda()
            #
            # x_all_tail = fgot_all_x[add_ones]
            # x_neg_all_tail = fgot_all_x_neg[add_ones]
            # y_all_tail = torch.ones((x_all_tail.shape[0])).cuda()
            #
            # # head
            # x_context_head = fgot_all_x[unadd_ones]
            # x_context_head = x_context_head[:max(x_context_head.shape[0]//2, 1)]
            # x_neg_context_head = fgot_all_x_neg[unadd_ones]
            # x_neg_context_head = x_neg_context_head[:max(x_neg_context_head.shape[0]//2, 1)]
            # y_context_head = torch.ones((x_context_head.shape[0])).cuda()
            #
            # x_all_head = fgot_all_x[unadd_ones]
            # x_neg_all_head = fgot_all_x_neg[unadd_ones]
            # y_all_head = torch.ones((x_all_head.shape[0])).cuda()
            #
            # # combining negative set:
            # x_context_tail = torch.cat((x_context_tail, x_neg_context_tail))
            # y_context_tail = torch.cat((y_context_tail, torch.zeros((y_context_tail.shape[0])).cuda()))
            # x_all_tail = torch.cat((x_all_tail, x_neg_all_tail)).cuda()
            # y_all_tail = torch.cat((y_all_tail, torch.zeros(y_all_tail.shape[0]).cuda()))
            #
            # z_tail, kld_tail, target_head = self.neural_processes(x_all_tail, y_all_tail, x_context_tail, y_context_tail)  # b, D
            #
            #
            # x_context_head = torch.cat((x_context_head, x_neg_context_head))
            # y_context_head = torch.cat((y_context_head, torch.zeros((y_context_head.shape[0])).cuda()))
            # x_all_head = torch.cat((x_all_head, x_neg_all_head)).cuda()
            # y_all_head = torch.cat((y_all_head, torch.zeros(y_all_head.shape[0]).cuda()))
            #
            # z_head, kld_head, target_tail = self.neural_processes(x_all_head, y_all_head, x_context_head, y_context_head)  # b, D

            # v4: unified indexing:
            fgot_all_x = torch.cat((sub_feats,feats), dim=-1)
            fgot_all_x_neg = torch.cat((sub_feats, feats[f]), dim=-1)

            # no_tail
            x_context = fgot_all_x
            x_context = x_context[:max(int(x_context.shape[0]*0.4), 1)]
            x_neg_context = fgot_all_x_neg
            x_neg_context = x_neg_context[:max(int(x_neg_context.shape[0]*0.4), 1)]
            y_context = torch.ones((x_context.shape[0])).cuda()

            x_all = fgot_all_x
            x_neg_all = fgot_all_x_neg
            y_all = torch.ones((x_all.shape[0])).cuda()

            x_context = torch.cat((x_context, x_neg_context))
            y_context = torch.cat((y_context, torch.zeros((y_context.shape[0])).cuda()))
            x_all = torch.cat((x_all, x_neg_all)).cuda()
            y_all = torch.cat((y_all, torch.zeros(y_all.shape[0]).cuda()))

            z, kld, target = self.neural_processes(x_all, y_all, x_context, y_context)  # b, D



            # calibration:
            tail_calitor_1 = self.feature_cali(feats[add_ones]+z)
            feats[add_ones] += tail_calitor_1

            head_calitor_1 = self.feature_cali(feats[unadd_ones]+z)
            feats[unadd_ones] += head_calitor_1

            # prototype:
            tail_proto = self.recon_cali(z)

            # reconn
            recon_score = tail_proto @ x_sub_down_wop.T
            values, indi = torch.sort(recon_score,dim=-1)
            rec_pre = x_sub_down_wop[indi.squeeze()] #  values.squeeze()[:,None] *
            rec_pre = rec_pre.unsqueeze(dim=0).permute(0, 2, 1)
            recon_score = self.conv(rec_pre).permute(0, 2, 1).squeeze()
            recnn = torch.zeros_like(recon_score).cuda()
            recnn[indi.squeeze()] += recon_score
            out = self.post_gnn(recnn, dadj_t, dptr)

            # if-1
            logits_np = classifier(out+feats).log_softmax(dim=1) #[train_idx]

            weight = torch.tensor([1/max((1-dy).sum().item(),1), 1/max(dy.sum().item(), 1)]).cuda()
            # loss = smoothed(logits_np, dy, selected, self.args.n_class) #[train_idx]
            loss = F.nll_loss(logits_np, dy, weight=weight)
            loss += kld.mean()
            # loss += kld_head.mean()
            # loss += kl_divergence(target_tail, target_head).sum(-1)
            # loss += hsic_regular(head_calitor_1, head_calitor_2)

            return loss, logits_np, dy

        else:
            H = encoder(dx, dadj_t, dptr)
            logits_np = classifier(H).log_softmax(dim=1)
            loss = F.nll_loss(logits_np, dy)
            return loss, logits_np, dy



    def eval_model(self, batch, encoder, classifier, switch="ours"):
        data, train_idx = batch['data'], batch['train_idx']
        if self.args.dataset != "REDDIT-BINARY":
            dx = data.x.cuda()
        else:
            dx = torch.ones((data.num_nodes, 1)).cuda()

        dadj_t = data.adj_t.cuda()
        dptr = data.ptr.cuda()
        dbatch = data.batch.cuda()
        dy = data.y.cuda()

        if switch == "ours":
            sub_mask, sub_edge_index = self.get_k_top_subgraphs(dadj_t, dbatch)

            x_down_1, x_down_wop = self.gnn_down_1(dx, dadj_t, dptr)
            feats = self.self_attn(x_down_1, self.down_param_k, self.down_param_k, need_weights=False)[0] # B D

            x_sub_down_1, x_sub_down_wop = self.gnn_down_1(dx, sub_edge_index, dptr, sub_mask)
            sub_feats = self.self_attn(x_sub_down_1, self.down_param_k, self.down_param_k, need_weights=False)[0] # B D

            f = [ix for ix in range(feats.shape[0])]
            random.shuffle(f)

            fgot_all_x = torch.cat((sub_feats,feats), dim=-1)
            fgot_all_x_neg = torch.cat((sub_feats, feats[f]), dim=-1)


            x_all = fgot_all_x
            y_all = torch.ones((fgot_all_x.shape[0])).cuda()

            # combining negative set:
            x_all = torch.cat((x_all, fgot_all_x_neg)).cuda()
            y_all = torch.cat((y_all, torch.zeros(fgot_all_x_neg.shape[0]).cuda()))

            z, _, _ = self.neural_processes(x_all, y_all)  # b, D

            # calibration:
            tail_calitor_1 = self.feature_cali(feats+z)
            feats += tail_calitor_1

            # prototype:
            tail_proto = self.recon_cali(z)

            # reconn
            recon_score = tail_proto @ x_sub_down_wop.T
            values, indi = torch.sort(recon_score, dim =-1)
            rec_pre =  x_sub_down_wop[indi.squeeze()] # values.squeeze()[:, None] *
            rec_pre = rec_pre.unsqueeze(dim=0).permute(0, 2, 1)
            recon_score = self.conv(rec_pre).permute(0, 2, 1).squeeze()
            recnn = torch.zeros_like(recon_score).cuda()
            recnn[indi.squeeze()] += recon_score
            out = self.post_gnn(recnn, dadj_t, dptr)

            # if-1
            logits_np = classifier(feats+out).log_softmax(dim=1)  # [train_idx]

            weight = torch.tensor([1/max((1-dy).sum().item(),1), 1/max(dy.sum().item(), 1)]).cuda()
            # loss = smoothed(logits_np, dy, selected, self.args.n_class) #[train_idx]
            loss = F.nll_loss(logits_np, dy, weight=weight)
            # loss += hsic_regular(z_2.detach(), deep_features[-self.moto:])

            return loss, logits_np, dy

        else:
            H = encoder(dx, dadj_t, dptr)
            logits_np = classifier(H).log_softmax(dim=1)
            loss = F.nll_loss(logits_np, dy)
            return loss, logits_np, dy


    def get_subgraphs(self, graph_batch):
        subs = []
        idxes = [graph.id for graph in graph_batch]
        for idx in idxes:
            picks = graph_batch[idx*self.core_nums: (idx+1)*self.core_nums]
            assert picks[0][0] == idx
            subs += [item[1] for item in picks]   # 展平

        return subs


    def neural_processes(self, fea_all, y_all, fea_context=None, y_context=None):
        '''

        Args:
            fea_context: b K D
            y_context: b
            fea_all: B K D
            y_all: B

        Returns: b K D

        '''
        if fea_context is not None and y_context is not None:
            context_r = self.latent_encoder(fea_context, y_context)
            all_r = self.latent_encoder(fea_all, y_all)
            context_dist = self.xy_to_mu_sigma(context_r)  # D
            target_dist = self.xy_to_mu_sigma(all_r) # D
            z = target_dist.rsample(sample_shape=(1, ))

            kld = kl_divergence(target_dist, context_dist).sum(-1)
            # z, kld = self.flows_2(z, target_dist, context_dist)
        else:
            all_r = self.latent_encoder(fea_all, y_all)
            target_dist = self.xy_to_mu_sigma(all_r)
            z = target_dist.sample(sample_shape=(1, ))

            # z, _ = self.flows_2(z, target_dist)
            kld = None
        # if self.np_flow != 'none':
        #     z, kld = self.flows(z, target_dist, context_dist)
        # else:
        #     kld = kl_divergence(target_dist, context_dist).sum(-1)
        return z, kld, target_dist



    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = torch_sparse.spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight
