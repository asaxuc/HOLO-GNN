import pickle

import numpy
import numpy as np
import torch

from utils import *
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


def get_headtail_mapping(data, whatasp, batch):  # class 这个还需要吗……
    if whatasp == "node":
        node_list = []
        for item in data:
            node_list.append(item.x.shape[0].item())
        m = sorted(node_list, reverse=True)
        ht_threshold = m[int(len(m) * 0.2)]

        temp = torch.zeros_like(batch.id)
        for idx, ite in enumerate(batch):  # (小的)
            if ite.shape[0].item() > ht_threshold:
                temp[idx] = 1
        return temp.to(torch.bool)


    elif whatasp == "degree":
        deg_list = []
        for item in data:
            ttte = torch.stack(item.adj_t.coo()[:2])
            deg_list.append(torch.sum(degree(ttte[-1])).item())
        m = sorted(deg_list, reverse=True)
        ht_threshold = m[int(len(m) * 0.2)]

        temp = torch.zeros_like(batch.id)
        for idx, ite in enumerate(batch):  # (小的)
            ttte = torch.stack(ite.adj_t.coo()[:2])
            if torch.sum(degree(ttte[-1])).item() > ht_threshold:
                temp[idx] = 1
        return temp.to(torch.bool)

# def train(encoder, classifier, data_loader, optimizer_e, optimizer_c, args):
#     encoder.train()
#     classifier.train()
#
#     total_loss = 0
#     for i, batch in enumerate(data_loader):
#         batch_to_gpu(batch, args.device)
#         data, train_idx = batch['data'], batch['train_idx']
#
#         knn_adj_t, aug_adj_ts, aug_xs = batch['knn_adj_t'], batch['aug_adj_ts'], batch['aug_xs']
#
#         H_augs, logit_aug_props = [], []
#
#         for i in range(args.aug_num):
#             H_augs.append(encoder(aug_xs[i], aug_adj_ts[i], data.ptr))
#
#             H_knn = H_augs[-1]
#             for k in range(args.knn_layer):
#                 H_knn = torch.sparse.mm(knn_adj_t, H_knn)
#             logit_aug_props.append(classifier(H_knn)[train_idx])
#
#         loss = 0
#         for i in range(args.aug_num):
#             loss += F.nll_loss(logit_aug_props[i], data.y[train_idx])
#         loss = loss / args.aug_num
#
#         loss = loss + consis_loss(logit_aug_props, temp=args.temp)
#
#         optimizer_e.zero_grad()
#         optimizer_c.zero_grad()
#         loss.backward()
#         optimizer_e.step()
#         optimizer_c.step()
#
#         total_loss += (loss * train_idx.shape[0]).item()
#
#     return total_loss / (i + 1)


# def train_baseline(learner, encoder, classifier, data_loader, optimizer_e, optimizer_c, args):
#     encoder.train()
#     classifier.train()
#
#     total_loss = 0
#     for i, batch in enumerate(data_loader):
#         batch_to_gpu(batch, args.device)
#         data, train_idx = batch['data'], batch['train_idx']
#
#         knn_adj_t, aug_adj_ts, aug_xs = batch['knn_adj_t'], batch['aug_adj_ts'], batch['aug_xs']
#
#         h = encoder(aug_xs[i], aug_adj_ts[i], data.ptr)
#         m = classifier(h)
#
#         loss += F.nll_loss(logit_aug_props[i], data.y[train_idx])
#         loss = loss / args.aug_num
#         loss = loss
#
#         optimizer_e.zero_grad()
#         optimizer_c.zero_grad()
#         loss.backward()
#         optimizer_e.step()
#         optimizer_c.step()
#
#         total_loss += (loss * train_idx.shape[0]).item()
#
#     return total_loss / (i + 1)

def xunlian(args, learner, encoder, classifier, data_loader, optimizer_e, optimizer_c):
    encoder.train()
    classifier.train()

    total_loss = 0
    pred = []
    truth = []
    for i, batch in enumerate(data_loader):
        # batch_to_gpu(batch, args.device)

        loss, logits, dy = learner(batch,encoder,classifier,  switch=args.method)

        optimizer_e.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer_e.step()
        optimizer_c.step()

        pred.extend(logits.argmax(-1).tolist())
        truth.extend(dy.tolist())

        total_loss += (loss).item()

    acc_c = f1_score(truth, pred, labels=np.arange(
        0, 2), average=None, zero_division=0)
    acc = (np.array(pred) == np.array(truth)).sum() / len(truth)

    # print("TRAIN: LOSS: {}, ACC: {}, F1-macro: {}, F1-micro: {}".format(total_loss / (len(data_loader) + 1), ((torch.tensor(pred) == torch.tensor(truth)).sum()/torch.tensor(truth).numel()).item(), np.mean(acc_c), acc))

    return {"T-ACC": ((torch.tensor(pred) == torch.tensor(truth)).sum()/torch.tensor(truth).numel()).item(), "T-F1-macro":np.mean(acc_c), "T-F1-micro":acc}


@torch.no_grad()
def eval(args, learner, encoder, classifier, data_loader, ptr=False, bdata=None, indi=None):
    learner.eval()
    encoder.eval()
    classifier.eval()

    total_loss = 0
    pred = []
    truth = []

    o_tail = []
    t_tail = []

    o_head = []
    t_head = []

    for i, batch in enumerate(data_loader):
        # batch_to_gpu(batch, args.device)
        das = batch["data"]
        loss, logits, dy = learner.eval_model(batch,encoder,classifier, args.method)

        pred.extend(logits.argmax(-1).tolist())
        truth.extend(dy.tolist())

        total_loss += (loss).item()

        pred_tail = torch.tensor(logits.argmax(-1).tolist())
        dy_tail = torch.tensor(dy.tolist())
        das = batch["data"]

        marker1 = torch.tensor(())

        # o_tail.extend(pred_tail[~get_headtail_mapping(bdata,"node",das)].numpy().tolist())
        # t_tail.extend(dy_tail[~get_headtail_mapping(bdata,"node",das)].numpy().tolist())


        o_tail.extend(pred_tail[dy_tail==0].numpy().tolist())
        t_tail.extend(dy_tail[dy_tail==0].numpy().tolist())

        o_head.extend(pred_tail[dy_tail==1].numpy().tolist())
        t_head.extend(dy_tail[dy_tail==1].numpy().tolist())


    acc_c =  f1_score(truth, pred, labels=np.arange( 0, 2), average=None, zero_division=0)
    acc =  (np.array(pred) == np.array(truth)).sum() / len(truth)


    acc_c_tail =  f1_score(t_tail, o_tail, labels=np.arange( 0, 2), average=None, zero_division=0)
    acc_tail = (np.array(o_tail) == np.array(t_tail)).sum() / (len(t_tail)+1)

    acc_c_head =   f1_score(t_head, o_head, labels=np.arange(0, 2), average=None, zero_division=0)
    acc_head = (np.array(o_head) == np.array(t_head)).sum() / (len(t_head)+1)


    if ptr:
        print("EVAL: LOSS: {}, ACC: {}, F1-macro: {}, F1-micro: {}".format(total_loss / (len(data_loader) + 1), ((torch.tensor(pred) == torch.tensor(truth)).sum()/torch.tensor(truth).numel()).item(), np.mean(acc_c), acc))


    # bestone1 = bestone
    # print(np.mean(acc_c) ,bestone[0] , acc , bestone[1])
    # if np.mean(acc_c) > bestone[0] and acc > bestone[1]:
    #     os.makedirs("./discussions/", exist_ok=True)
    #     sl = {"data": datas, "pred": preds, "truth": truths}
    #     with open("./discussions/dis.pkl", "wb") as f:
    #         pickle.dump(sl, f)
    #     f.close()
    bestone1 = [np.mean(acc_c),acc]
    return {"loss":total_loss / (len(data_loader) + 1), "ACC":((torch.tensor(pred) == torch.tensor(truth)).sum()/torch.tensor(truth).numel()).item(), "F1-macro":np.mean(acc_c),
            "F1-micro":acc,"F1-macro-tail":np.mean(acc_c_tail),"F1-micro-tail":np.mean(acc_tail), "F1-macro-head":np.mean(acc_c_head),"F1-micro-head":np.mean(acc_head)},bestone1

    # return {"loss":total_loss / (len(data_loader) + 1), "ACC":((torch.tensor(pred) == torch.tensor(truth)).sum()/torch.tensor(truth).numel()).item(), "F1-macro":np.mean(acc_c), "F1-micro":acc,"ACC_TAIL":acc_tail,"ACC_HEAD":acc_head }
