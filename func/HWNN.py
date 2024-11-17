#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:HWNN
@author:xiangguosun
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: HWNN.py
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1
@time: 2019/10/16
"""
import os
import torch
import torch.nn.functional as F
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
import argparse




class HWNNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ncount,  K1=2, K2=2, approx=False):
        super(HWNNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncount = ncount
        self.device = torch.device("cuda")
        self.K1 = K1
        self.K2 = K2
        self.approx = approx
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(self.ncount))
        self.par = torch.nn.Parameter(torch.Tensor(self.K1 + self.K2))
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.99, 1.01)
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def forward(self, features, snap_index, data):
        # features: (n, d)
        # diagonal : n
        diagonal_weight_filter = torch.diag(self.diagonal_weight_filter).to(self.device)
        features = features.to(self.device)
        # Theta=self.data.Theta.to(self.device)
        Theta = data["Theta"].to(self.device)  # 起码，会给一个 （n e）。 当然可以算一下
        Theta_t = torch.transpose(Theta, 0, 1)

        if self.approx:
            poly = self.par[0] * torch.eye(self.ncount).to(self.device)
            Theta_mul = torch.eye(self.ncount).to(self.device)
            for ind in range(1, self.K1):
                Theta_mul = Theta_mul @ Theta
                poly = poly + self.par[ind] * Theta_mul

            poly_t = self.par[self.K1] * torch.eye(self.ncount).to(self.device)
            Theta_mul = torch.eye(self.ncount).to(self.device)
            for ind in range(self.K1 + 1, self.K1 + self.K2):
                Theta_mul = Theta_mul @ Theta_t  # 这里也可以使用Theta_transpose
                poly_t = poly_t + self.par[ind] * Theta_mul

            # poly=self.par[0]*torch.eye(self.ncount).to(self.device)+self.par[1]*Theta+self.par[2]*Theta@Theta
            # poly_t = self.par[3] * torch.eye(self.ncount).to(self.device) + self.par[4] * Theta_t + self.par[5] * Theta_t @ Theta_t
            # poly_t = self.par[3] * torch.eye(self.ncount).to(self.device) + self.par[4] * Theta + self.par[
            #     5] * Theta @ Theta
            local_fea_1 = poly @ diagonal_weight_filter @ poly_t @ features @ self.weight_matrix
        else:
            print("wavelets!")
            wavelets = data["wavelets"].to(self.device)
            wavelets_inverse = self.data["wavelets_inv"].to(self.device)
            local_fea_1 = wavelets @ diagonal_weight_filter @ wavelets_inverse @ features @ self.weight_matrix

        localized_features = local_fea_1
        return localized_features


class HWNN(torch.nn.Module):
    def __init__(self, ncount, n_hyper, feature_number):
        super(HWNN, self).__init__()
        self.filters = 128
        # self.features=features
        self.ncount = ncount
        self.feature_number = feature_number

        self.hyper_snapshot_num = n_hyper
        print("there are {} hypergraphs".format(self.hyper_snapshot_num))

        self.convolution_1 = HWNNLayer(self.feature_number,
                                       self.filters,
                                       self.ncount,
                                       K1=3,
                                       K2=3,
                                       approx=True )

        self.convolution_2 = HWNNLayer(self.filters,
                                       self.filters,
                                       self.ncount,
                                       K1=3,
                                       K2=3,
                                       approx=True )

        self.par = torch.nn.Parameter(torch.Tensor(self.hyper_snapshot_num))
        torch.nn.init.uniform_(self.par, 0, 0.99)  # 1.0)



    def forward(self, features):
        features = features.to(self.device)
        channel_feature = []
        for snap_index in range(self.hyper_snapshot_num):
            deep_features_1 = F.relu(self.convolution_1(features,
                                                        snap_index,
                                                        self.data))
            deep_features_1 = F.dropout(deep_features_1, self.args.dropout)
            deep_features_2 = self.convolution_2(deep_features_1,
                                                 snap_index,
                                                 self.data)
            deep_features_2 = F.log_softmax(deep_features_2, dim=1)  # 把这里换成relu会怎么样呢？
            channel_feature.append(deep_features_2)

        deep_features_3 = torch.zeros_like(channel_feature[0])
        for ind in range(self.hyper_snapshot_num):
            deep_features_3 = deep_features_3 + self.par[ind] * channel_feature[ind]

        return deep_features_3