import itertools
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from numpy import test
from parse import parse_args
from tqdm import tqdm
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
import math
from utils import *
from learn import *
from dataset import *
from dataprocess import *
from func import *
from func.flow import MuSigmaEncoder, LatentEncoder

def run(args):
    TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S")
    if os.path.exists(f"./log/{args.dataset}_{args.method}_{TIMESTAMP}"):
        shutil.rmtree(f"./log/{args.dataset}_{args.method}_{TIMESTAMP}")
    writer = SummaryWriter(f"./log/{args.dataset}_{args.method}_{TIMESTAMP}")

    pbar = tqdm(range(args.runs), unit='run')

    F1_micro = np.zeros(args.runs, dtype=float)
    F1_macro = np.zeros(args.runs, dtype=float)

    F1_micro_head = np.zeros(args.runs, dtype=float)
    F1_macro_head = np.zeros(args.runs, dtype=float)

    F1_micro_tail = np.zeros(args.runs, dtype=float)
    F1_macro_tail = np.zeros(args.runs, dtype=float)

    for count in pbar:
        random.seed(args.seed + count)
        np.random.seed(args.seed + count)
        torch.manual_seed(args.seed + count)
        torch.cuda.manual_seed_all(args.seed + count)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # rand_node = torch.rand(args.degb.shape)
        # rand_edge = torch.rand(args.degb.shape)
        # args.aug_edge = rand_edge <= args.p_edge
        # args.aug_node = rand_node <= args.p_node

        train_data, val_data, test_data = shuffle(
            dataset, args.c_train_num, args.c_val_num, args.y)

        train_data = upsample(train_data)
        val_data = upsample(val_data)
        # if not args.use:
        #     train_data = upsample(train_data)
        #     val_data = upsample(val_data)
        # else:
        # train_data, ext_data = construct_context_query(train_data)  # 内部多采样一点。
        # val_data, ext_data = construct_context_query(val_data)

        # for batch in train_data:
        #     for key in batch.keys():
        #         if isinstance(batch[key], list):
        #             for i in range(len(batch[key])):
        #                 batch[key][i] = batch[key][i].cuda()
        #         else:
        #             batch[key] = batch[key].cuda()

        query_nums = args.batch_size
        assert query_nums == args.batch_size
        train_dataset = Dataset_NP(train_data, dataset, args )
        val_dataset = Dataset_NP(val_data, dataset, args )
        test_dataset = Dataset_NP(test_data, dataset, args )

        temp = ImbSampler(train_dataset,args)
        args.heads = temp.head_classe_index
        args.tails = temp.tail_classe_index

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, collate_fn=train_dataset.collate_batch)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=val_dataset.collate_batch)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=test_dataset.collate_batch)

        # stripped
        encoder = GIN(args, args.n_feat, args.n_hidden).to(args.device)

        classifier = MLP_Classifier(args).to(args.device)

        learner = Learner(args, 20, 10, query_nums ).to(args.device)


        p1 = []
        p2 = []
        for name, params in learner.named_parameters():
            if "xy_to_mu_sigma" in name or "latent_encoder" in name:
                p1.append(params)
            else:
                p2.append(params)
        lla = [{'params': p1, 'lr': args.lr  }, {'params': p2, 'lr': args.lr}]

        optimizer_e = torch.optim.Adam(lla, lr=args.lr, weight_decay=args.weight_decay)
        optimizer_c = torch.optim.Adam(
            itertools.chain(classifier.parameters(), encoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)

        best_val_mac = 0
        best_val_mic = 0
        best_val_loss = math.inf
        val_loss_hist = []
        test_eval1 = None
        for epoch in  range(0, args.epochs)  :
            train_eval = xunlian(args, learner, encoder, classifier, train_loader,
                           optimizer_e, optimizer_c )

            best_one = [0,0]
            val_eval,_ = eval(args, learner, encoder, classifier, val_loader, bdata=test_dataset )
            test_eval,_ = eval(args, learner, encoder, classifier, test_loader, bdata=test_dataset )
            print(test_eval)

            if test_eval['F1-macro'] > best_val_mac and val_eval['F1-micro'] > best_val_mic:
                best_val_mac = test_eval['F1-macro']
                best_val_mic = test_eval['F1-micro']


            if (val_eval['loss'] < best_val_loss):
                best_val_loss = val_eval['loss']
                test_eval1 = test_eval


            val_loss_hist.append(val_eval['loss'])

            if count == 0:
                forw = {**train_eval, **test_eval}
                a = forw["loss"]
                forw.pop("loss")
                writer.add_scalars("NL3_acc_change", forw , epoch)
                writer.add_scalars("NL3_loss_change", {"loss":a} , epoch)

            if (args.early_stopping > 0 and epoch > args.epochs // 2):
                tmp = torch.tensor(
                    val_loss_hist[-(args.early_stopping + 1): -1])
                if (val_eval['loss'] > tmp.mean().item()):
                    break

        F1_micro[count] = test_eval1['F1-micro']
        F1_macro[count] = test_eval1['F1-macro']

        F1_micro_head[count] = test_eval1['F1-micro-head']
        F1_macro_head[count] = test_eval1['F1-macro-head']

        F1_micro_tail[count] = test_eval1['F1-micro-tail']
        F1_macro_tail[count] = test_eval1['F1-macro-tail']

        print(F1_micro, F1_macro)

    return F1_micro, F1_macro, [F1_micro_head,F1_macro_head,F1_micro_tail,F1_macro_tail]


if __name__ == '__main__':
    args = parse_args()

    if args.dataset == "PTC_MR":
        args.num_train = args.num_val = 90
    elif args.dataset == "REDDIT-BINARY":
        args.num_train = args.num_val = 500
    elif args.dataset == "NCI1" or args.dataset == "NCI109":
        args.num_train = args.num_val = 1000
    elif args.dataset == "DHFR":
        args.num_train = args.num_val = 120
    elif args.dataset == "MUTAG":
        args.num_train = args.num_val = 100
    else:
        args.num_train = args.num_val = 300

    torch.cuda.set_device(args.gpu)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()
    
    dataset, args.n_feat, args.n_class, _ = get_TUDataset(
        args.dataset, pre_transform=T.ToSparseTensor())

    yyl = []
    sizel = []
    for data in dataset:
        if args.dataset == "REDDIT-BINARY":
            data.x = torch.ones((data.num_nodes, 1))
        yyl.append(data.y.item())
        sizel.append(data.x.shape[0])

    args.c_train_num, args.c_val_num = get_class_num(
        args.imb_ratio, args.num_train, args.num_val)

    args.y = torch.tensor(yyl) # torch.tensor([data.y.item() for data in dataset])
    size_lst = sizel # [data.x.shape[0] for data in dataset]
    args.size = torch.tensor(size_lst)
    size_lst.sort()
    args.mid = size_lst[int(args.size_ratio * len(size_lst))]
    args.sizeb = args.size >= args.mid

    # args.degs = []
    # for data in dataset:
    #     edge_index = torch.stack(data.adj_t.coo()[:2])
    #     row, col = edge_index
    #     deg = degree(col)
    #     args.degs.append(deg.mean().item())
    # degs = sorted(args.degs)
    # args.middeg = degs[int(args.deg_ratio * len(degs))]
    # args.degs = torch.tensor(args.degs)
    # args.degb = args.degs > args.middeg
    # scal = np.log(degs[-1] / degs[0])
    # args.p_edge = (args.aug_ratio - (1 - args.aug_ratio)) * torch.log(args.degs / degs[0]) / scal + 1 - args.aug_ratio
    # args.p_node = ((1 - args.aug_ratio) - args.aug_ratio) * torch.log(args.degs / degs[0]) / scal + args.aug_ratio

    # args.kernel_idx, args.knn_edge_index = get_kernel_knn(args.dataset, args.kernel_type, args.knn_nei_num, args.sizeb)


    F1_micro, F1_macro, li = run(args)

    print('F1_macro: ', np.mean(F1_macro))
    print('F1_micro: ', np.mean(F1_micro))

    print('head and tail: ', [np.mean(i) for i in li])
