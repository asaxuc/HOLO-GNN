import os
import pickle
import random
from dataset import get_TUDataset
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.utils import subgraph, degree


def get_subgraphs(name, graph_batch, core_nums, hop_range):
    subs = []
    for idxx, graph_one in tqdm(enumerate(graph_batch)):
        nodes_index = torch.unique(torch.stack(graph_one.adj_t.coo()[:2]))
        ido = graph_one.id[0].item()
        deg1 = degree(graph_one.adj_t.coo()[1])

        if len(nodes_index.cpu().numpy().tolist()) > core_nums:
            cores = random.sample(nodes_index.cpu().numpy().tolist(),  core_nums)
        else:
            cores = torch.multinomial( (deg1.float()+1)**(1/2), core_nums, replacement=True)

        for n in cores:
            temp = torch.tensor(graph_one.adj_t.to_torch_sparse_coo_tensor().coalesce().indices())
            reg_node = []
            reg_node.append(n)
            for hop in range(hop_range):
                for idx in range(temp.shape[-1]):
                    if temp[0, idx] == n:
                        reg_node.append(temp[1, idx])
                    elif temp[1, idx] == n:
                        reg_node.append(temp[0, idx])
            print(len(reg_node)/torch.unique(temp).shape[0])
            sub,_ = subgraph(reg_node, temp)
            subs.append([ido, sub])

        if idxx % 20 == 0:
            os.makedirs(f"./data/{name}",exist_ok=True)
            with open(f"./data/{name}/{name}.pkl","wb") as f:
                pickle.dump(subs, f)

    return subs



def handler(dataset_name):
    dataset = get_TUDataset(dataset_name, pre_transform=T.ToSparseTensor())
    get_subgraphs(dataset_name, dataset[0], 20, 2)

# handler("DD") # hop-5
# handler("PROTEINS") # hop-3
# handler("REDDIT-BINARY") # hop- x
# handler("PTC_MR") # hop- 1
handler("NCI1") # hop- 2


# name = "DD"
# with open(f"./data/{name}/{name}.pkl", "rb") as f:
#    data = pickle.load(f)
# print(data)
# def origin_writedown(name, dataset):
#     ini = [0,0]
#     y_num = [0,0]
#     dataset = dataset[0]
#     for graph in dataset:
#         y_num[graph.y[0].item()] += 1
#
#     os.makedirs(f"./data/{name}", exist_ok=True)
#     acc = 0
#     with open(f"./data/{name}/{name}.txt","w") as f:
#         f.write(str(len(dataset))+'\n')
#         for graph in dataset:
#             if graph.y[0].item() == 0 and ini[0] == 0 :
#                 f.write(str(y_num[0])+' 0\n')
#                 ini[0] = 1
#             elif graph.y[0].item() == 1 and ini[1] == 0:
#                 f.write(str(y_num[1])+' 1\n')
#                 ini[1] = 1
#
#             graph_nodes = torch.unique(torch.stack(graph.adj_t.coo()[:2])).numpy().tolist()
#             for idx, item in enumerate(graph_nodes):
#                 if idx != len(graph_nodes) -1:
#                     f.write(str(acc+item)+' ')
#                 else:
#                     f.write(str(acc+item)+'\n')
#             acc += len(graph_nodes)
#
# handler("DD")



# def subgraph_sample(dataset, graph_list, border, nums = 500):
#     with open('dataset/%s/sampling.txt' % (dataset), 'w') as f:
#         f.write(str(len(graph_list))+'\n')
#         for gra in graph_list:
#             graph = {}
#             graph.sample_list = []
#             graph.unsample_list = []
#             graph.sample_x = []
#             n = gra.g.number_of_nodes()
#             K = int(min(border-1, n/2))
#             f.write(str(K)+'\n')
#             graph.K = K
#             for i in range(nums):
#                 sample_idx = np.random.permutation(n)
#                 j = 0
#                 sample_set = set()
#                 wait_set = []
#                 cnt = 0
#                 if (len(graph.neighbors[j]) == 0):
#                     j += 1
#                 wait_set.append(sample_idx[j])
#                 while cnt < K:
#                     if len(wait_set) != 0:
#                         x = wait_set.pop()
#                     else:
#                         break
#                     while x in sample_set:
#                         if len(wait_set) != 0:
#                             x = wait_set.pop()
#                         else:
#                             cnt = K
#                             break
#                     sample_set.add(x)
#                     cnt += 1
#                     wait_set.extend(graph.neighbors[x])
#                 unsample_set = set(range(n)).difference(sample_set)
#                 f.write(str(len(sample_set)) + ' ')
#                 for x in list(sample_set):
#                     f.write(str(x) + ' ')
#                 for x in list(unsample_set):
#                     f.write(str(x) + ' ')
#                 f.write('\n')