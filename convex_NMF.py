#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script only implement the CNMF as in paper 
    Convex and Semi-Nonnegative Matrix Factorizations, Chris Ding et,al.

will be used in the initialization step of online method, to get W_0, and cluster X_0 into 
X_hat, which has clusters assignment of each data sample.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import ipdb
import os
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
# from cvx_online_NMF import initialize_X_W_hat
from common_functions import kmeans_clustering

def CNMF_algo(X, W_0, G_0, eps = 1e-5, max_iter = 500000):
    '''
    X = XW @ G.T
    '''
    W_old,  G_old = W_0.copy(), G_0.copy()
    err = lambda _x, _y: np.linalg.norm(X - X @ _x @ _y.T)
    pos = lambda mat: np.clip(mat, 0, None)
    neg = lambda mat: pos(-1 * mat)
    X_norm = X.T @ X
    X_pos = pos(X_norm)
    X_neg = neg(X_norm)

    k = 0

    while err(W_old, G_old) > eps and k < max_iter:
        XW_pos = X_pos @ W_old
        XW_neg = X_neg @ W_old
        common_1 = G_old @ W_old.T
        # T1 = X_pos @ W_old + G_old @ W_old.T @ X_neg @ W_old
        # T2 = X_neg @ W_old + G_old @ W_old.T @ X_pos @ W_old
        T1 = XW_pos + common_1 @ XW_neg
        T2 = XW_neg + common_1 @ XW_pos
        G_new = G_old * np.sqrt( T1 / T2)
        G_old = G_new

        # ---------
        XG_pos = X_pos @ G_old
        XG_neg = X_neg @ G_old
        common_2 = G_old.T @ G_old
        # T3 = X_pos @ G_old + X_neg @ W_old @ G_old.T @ G_old
        # T4 = X_neg @ G_old + X_pos @ W_old @ G_old.T @ G_old
        T3 = XG_pos + XW_neg @ common_2
        T4 = XG_neg + XW_pos @ common_2
        W_new = W_old * np.sqrt(T3 / T4)
        W_old = W_new

        k += 1

    return W_new, G_new

def CNMF_init(X_0, k_cluster):
    m_dim, n_dim = X_0.shape
    kmeans = KMeans(n_clusters = k_cluster, max_iter = 1000)
    kmeans.fit(X_0.T)
    X_hat_assignments = kmeans.labels_

    cluster_count = [len(np.where(X_hat_assignments == i)[0]) for i in range(k_cluster)]
    
    D = np.zeros((k_cluster, k_cluster), int)
    for idx in range(k_cluster):
        D[idx][idx] = cluster_count[idx] + 1e-3

    H = np.zeros((n_dim, k_cluster), int)
    for idx in range(k_cluster):
        non_zero_idx = np.where(X_hat_assignments == idx)[0]
        H[non_zero_idx, idx] = 1
    
    W_init = np.matmul((H + np.ones(H.shape, int) * 0.2), np.linalg.inv(D))
    G_init = H + 0.2
    return W_init, G_init

def CNMF(X_0, k_cluster, eps = 1e-5, max_iter = 5e5):
    '''
    X_0 = [x_1, ..., x_n], column vector being data
    X = X @ W @ G.T
    input:
        X_0: m_dim * n_dim
        k_cluster
    return:
        W_new: n_dim * k_cluster
        G_new: n_dim * k_cluster
    '''
    W_0, G_0 = CNMF_init(X_0, k_cluster)

    W_new, G_new = CNMF_algo(X_0, W_0, G_0, eps, max_iter)
    
    _, assignment = kmeans_clustering(G_new, k_cluster, 1200)
    return W_new, G_new, assignment

if __name__ == '__main__':
    n_dim, m_dim = 10, 2
    X_0 = np.random.randn(n_dim, m_dim).T
    k_cluster = 5

    W_new, G_new, assignment = CNMF(X_0, k_cluster)
    # print(W_new)
    # print(G_new)
    print(assignment)
    print(np.linalg.norm(X_0))
    print(np.linalg.norm(X_0 - X_0 @ W_new @ G_new.T))

