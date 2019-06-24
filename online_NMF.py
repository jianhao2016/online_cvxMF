#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This is an implementation of the paper online NMF.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import pdb
import os
import argparse
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE

def online_dict_learning(X, lmda, D_0, T, k_cluster, eps, _NF = 200):
    '''
    algo 1 in the paper
    D_0: R^(m * k)
    X: R^(n * m)
    '''
    n_dim, m_dim = X.shape
    A_t = np.zeros((k_cluster, k_cluster))
    B_t = np.zeros((m_dim, k_cluster))
    D_t = D_0
    
    t_start = time.time()
    # print(lmda, _NF, eps)
    for t in range(T):
        # t_start_online = time.time()
        sample_idx = np.random.randint(0, n_dim)
        x_sample = X[sample_idx, :]

        lars_lasso = LassoLars(alpha = lmda)
        lars_lasso.fit(D_t, x_sample)
        alpha_t = lars_lasso.coef_

        A_t += np.matmul(alpha_t.reshape(k_cluster, 1), alpha_t.reshape(1, k_cluster))
        B_t += np.matmul(x_sample.reshape(m_dim, 1), alpha_t.reshape(1, k_cluster))

        D_t = dict_update(D_t, A_t, B_t, eps = eps, _NF = _NF)
        # print('===== Iteration in online dictionary learning cost {:.04f}s'.format(time.time() - t_start_online))
    print('Dcitionary update done! Time elapse {:.04f}s'.format(time.time() - t_start))
    return D_t


def dict_update(D_t, A_t, B_t, eps, _NF = 200):
    '''
    D_t: R^(m * k)
    A_t: R^(k * k)
    B_t: R^(m * k)
    '''
    m_dim, k_cluster = D_t.shape
    D_new = D_t.copy()

    # t_start = time.time()
    while True:
        D_old = D_new.copy()
        for j in range(k_cluster):
            grad = (B_t[:, j] - np.matmul(D_new, A_t[:, j]))
            u_j =  1/(A_t[j, j] + 1e-5) * grad + D_new[:, j]
            D_new[:, j] = (u_j / max(np.linalg.norm(u_j), 1)) * _NF
        if (np.linalg.norm(D_new - D_old) < eps):
            break
    # print('Iteration in dictionary update cost {:.04f}s'.format(time.time() - t_start))

    return D_new

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--it', type=int, default=600)
    parser.add_argument('--lmda', type=float, default=1e-1)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--normal_factor', '--NF', type=float, default=200)
    args = parser.parse_args()

    # set number of iteration, lambda in lasso, epsilon in dictionary update and normalization factor
    print(args)
    numIter = args.it
    lmda = args.lmda
    eps = args.eps
    _NF = args.normal_factor

    # np.random.seed(42)
    data_root = '/data/jianhao/scRNA_seq'
    df_file = os.path.join(data_root, 'pandas_dataframe')
    feat_file = os.path.join(data_root, 'df_feature_column')

    df = pd.read_pickle(df_file)
    with open(feat_file, 'rb') as f:
        feat_cols = pickle.load(f)
    X_raw = df[feat_cols].values
    X = normalize(X_raw) * _NF
    # X = X_raw
    Y = df['label'].values

    k_cluster = 3
    n_dim, m_dim = X.shape
    D_0 = np.random.randn(m_dim, k_cluster)
    D_0 = np.absolute(D_0)
    D_0 = normalize(D_0, axis = 0) * _NF
    print('D_0 = ', D_0)
    # pdb.set_trace()
    D_final = online_dict_learning(X, lmda = lmda, D_0 = D_0, T = numIter, k_cluster = 3, eps = eps)
    print('D_final = ', D_final)
    print('shape of D mat:', D_final.shape)
    print('norm of D_final', np.linalg.norm(D_final[:, 1]), np.linalg.norm(D_final[:, 2]))

    df_centroids = pd.DataFrame(D_final.T, columns = feat_cols)
    df_centroids['label'] = ['cell type {}'.format(x) for x in range(1, k_cluster + 1)]
    print('shape of centroid df:', df_centroids.shape)

    # print(df_centroids['label'].values)
    # print(np.max(df_centroids[feat_cols].values))
    # print(np.min(df_centroids[feat_cols].values))
    print('is D_centroids finite?', np.isfinite(df_centroids[feat_cols].values).all())

    # we need to normalize the input data X
    df_final = df.copy()
    # df_final[feat_cols] = normalize(X_raw)
    df_final[feat_cols] = X
    df_final = df_final.append(df_centroids)
    print('shape of df_final: ', df_final.shape)
    
    rndperm = np.random.permutation(df_final.shape[0])
    n_sne = 303

    t_start = time.time()
    tsne = TSNE(n_components = 2, verbose = 1, perplexity=40, n_iter = 1000)
    tsne_result = tsne.fit_transform(df_final[feat_cols].values)
    print('tsne_result shape = ', tsne_result.shape)

    print('t-SNE done! Time elapse {:.04f}s'.format(time.time() - t_start))
    df_tsne = df_final.copy()
    df_tsne['x-tsne'] = tsne_result[:, 0]
    df_tsne['y-tsne'] = tsne_result[:, 1]

    df_raw_tsne = df_tsne[:300]
    df_centroid_tsne = df_tsne[-3:]
    # print(df_raw_tsne.shape)
    # print(df_centroid_tsne.shape)
    
    sc_x = df_tsne['x-tsne'].values[:300]
    sc_y = df_tsne['y-tsne'].values[:300]
    sc_types = df_tsne['label'].values[:300]

    centroid_x = df_tsne['x-tsne'].values[-3:]
    centroid_y = df_tsne['y-tsne'].values[-3:]
    centroid_types = df_tsne['label'].values[-3:]
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    color_list = ['red', 'green', 'blue']

    for idx in range(3):
        ax.scatter(sc_x[idx*100:(idx+1)*100], sc_y[idx*100:(idx+1)*100], color = color_list[idx],
                label = sc_types[idx], s = 50, alpha = 0.6)


    ax2 = fig.add_subplot(122, sharex=ax, sharey=ax)
    marker_list = ['D', 's', '+']
    centroid_color_list = ['yellow', 'black', 'magenta']
    for idx in range(k_cluster):
        ax2.scatter(centroid_x[idx], centroid_y[idx], color = centroid_color_list[idx], marker = marker_list[idx],
                alpha = 0.8, label = centroid_types[idx])
    
    # ax.legend()
    fig.suptitle('tSNE of online NMF on scRNA expression\nwith eps = {}, lambda = {}, iter = {}'.format(eps, lmda, numIter))
    ax.set_xlabel('tSNE element 1')
    ax.set_ylabel('tSNE element 2')

    ax.grid(color = 'grey', alpha = 0.4)
    p2f = os.path.join(data_root, 'pic', 'eps_{:.0e}_lambda_{:.0e}_iteration_{:.0f}_Normal_{:.0f}'.format(eps, lmda, numIter, _NF))
    fig.savefig(p2f, dpi = 500)


