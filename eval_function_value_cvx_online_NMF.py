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
import ipdb
import os
import argparse
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.linalg as linalg
# import cvxpy as cvx
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score
from cluster_assignment_method import get_clustering_assignment_1, get_clustering_assignment_2
from common_functions import get_g_hat_value, evaluation_clustering
from common_functions import eval_g_hat_with_DnX, kmeans_clustering
from common_functions import geo_projection_to_cvx_cmb
from pycuda_update_W_comparison import update_W_hat_skcuda, opt_cal_W_hat_numpy
from pycuda_update_W_comparison import opt_cal_W_hat_solve, update_W_hat_numpy
from online_NMF import dict_update
from convex_NMF import CNMF_init
from visualization_NMF import plot_diff_method

def takespread(sequence, num):
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int((i * length / num))]

def my_normalize(X):
    '''
    scale X to be in a unit ball
    X: n x m, n samples in row
    '''
    n_dim, m_dim = X.shape
    max_norm = max([np.linalg.norm(X[k, :]) for k in range(n_dim)])
    X_new = X / max_norm
    return X_new

def my_CNMF(X_0, k_cluster, t_lower_bound, eps = 1e-5, max_iter = 5e5):
    '''
    X_0 is a column vector! Hence first transpose.
    X = X @ W @ G.T
    D = X @ W: m_dim * k_cluster.
    '''
    W_0, G_0 = CNMF_init(X_0, k_cluster)

    m_dim, n_dim = X_0.shape
    
    W_old,  G_old = W_0.copy(), G_0.copy()
    X = X_0
    err = lambda _x, _y: np.linalg.norm(X - X @ _x @ _y.T)
    pos = lambda mat: np.clip(mat, 0, None)
    neg = lambda mat: pos(-1 * mat)
    X_norm = X.T @ X
    X_pos = pos(X_norm)
    X_neg = neg(X_norm)

    k = 0
    t_cur = 0
    error_list_cmf = []
    t_end = time.time()
    t_start = time.time()

    while (err(W_old, G_old) > eps and k < max_iter) or t_end - t_start < t_lower_bound:
        D_t = X @ W_old
        error_t = eval_g_hat_with_DnX(X.T, D_t.T, n_dim, m_dim)
        # error_t = 1/n_dim * np.linalg.norm(X - D_t @ W_old.T)
        error_list_cmf.append((t_cur, error_t))
        
        t1 = time.time()
        XW_pos = X_pos @ W_old
        XW_neg = X_neg @ W_old
        common_1 = G_old @ W_old.T
        T1 = XW_pos + common_1 @ XW_neg
        T2 = XW_neg + common_1 @ XW_pos
        G_new = G_old * np.sqrt( T1 / T2)
        G_old = G_new

        # ---------
        XG_pos = X_pos @ G_old
        XG_neg = X_neg @ G_old
        common_2 = G_old.T @ G_old
        T3 = XG_pos + XW_neg @ common_2
        T4 = XG_neg + XW_pos @ common_2
        W_new = W_old * np.sqrt(T3 / T4)
        W_old = W_new
        t2 = time.time()
        t_cur += (t2 - t1)

        k += 1
        t_end = t_start + t_cur

    _, assignment = kmeans_clustering(G_new, k_cluster, 1200)

    return W_new, G_new, assignment, error_list_cmf


def my_online_dict_learning(X, lmda, D_0, T, k_cluster, 
        t_lower_bound, eps, _NF = 200):
    '''
    algo 1 in the paper
    D_0: R^(m * k)
    X: R^(n * m)
    '''
    n_dim, m_dim = X.shape
    A_t = np.zeros((k_cluster, k_cluster))
    B_t = np.zeros((m_dim, k_cluster))
    D_t = D_0
    
    t_end = time.time()
    t_start = time.time()
    t_cur = 0
    error_list_omf = []
    # print(lmda, _NF, eps)
    while t_end - t_start < t_lower_bound:
        for t in range(T):
            # t_start_online = time.time()
            error_t = eval_g_hat_with_DnX(X, D_t.T, n_dim, m_dim)
            error_list_omf.append((t_cur, error_t))
            t1 = time.time()
            sample_idx = np.random.randint(0, n_dim)
            x_sample = X[sample_idx, :]

            lars_lasso = LassoLars(alpha = lmda)
            lars_lasso.fit(D_t, x_sample)
            alpha_t = lars_lasso.coef_

            A_t += np.matmul(alpha_t.reshape(k_cluster, 1), alpha_t.reshape(1, k_cluster))
            B_t += np.matmul(x_sample.reshape(m_dim, 1), alpha_t.reshape(1, k_cluster))

            D_t = dict_update(D_t, A_t, B_t, eps = eps, _NF = _NF)
            t2 = time.time()
            t_cur += (t2 - t1)
            # print('===== Iteration in online dictionary learning cost {:.04f}s'.format(time.time() - t_start_online))
        t_end = t_start + t_cur
        # print('Dcitionary update done! Time elapse {:.04f}s'.format(time.time() - t_start))
    return D_t, error_list_omf

def my_nmf_clustering(X, k_cluster, numIter, t_lower_bound):
    ''' 
    X: n x m, where n is # of sample and n is sample dimension
        i.e. each row of X is a sample
    return:
    D: k x m, i.e. centroid in rows.
    X - weight * centroid
    '''
    n_dim, m_dim = X.shape

    centroid_matrix = np.abs(np.random.randn(k_cluster, m_dim))
    weight_matrix = np.abs(np.random.randn(n_dim, k_cluster))

    c_old = centroid_matrix
    w_old = weight_matrix
    t_cur = 0
    error_list_nmf = []
    t_end = time.time()
    t_start = time.time()

    while t_end - t_start < t_lower_bound:
        for it in range(numIter):
            error_it = eval_g_hat_with_DnX(X, centroid_matrix, n_dim, m_dim)
            error_list_nmf.append((t_cur, error_it))

            t1 = time.time()
            weight_matrix = w_old * ((X @ c_old.T)/ (w_old @ c_old @ c_old.T))
            w_old = weight_matrix
            centroid_matrix = c_old * ((w_old.T @ X) / (w_old.T @ w_old @ c_old))
            c_old = centroid_matrix
            t2 = time.time()
            t_cur += (t2 - t1)
        t_end = t_start + t_cur

    # err = np.linalg.norm(X - weight_matrix @ centroid_matrix)
    # print('l2 error = {:.4f}'.format(err))

    _, assignment = kmeans_clustering(weight_matrix, k_cluster, numIter)
    return centroid_matrix, weight_matrix, assignment, error_list_nmf


def cvx_online_dict_learning(X, y_true, n_hat, k_cluster, T, lmda, eps, 
        flag=True, version = 'Rr'):
    '''
    X: R^(n * m)
    y_true: str^n
    W_0: R^(n_hat * k)
    x_i : R^m
    alpha: R^k
    cvx_online problem 
        min||x_i - X.T * W * alpha|| + lambda * ||alpha||

    in the online setting, there is no X in (n * m), 
    instead, we need to store a candidate set and solve the subproblem:
        min ||x_i - X_hat * W_hat * alpha|| + lambda * ||alpha||

    X_hat : R^(m * n_hat)
    W_hat : R^(n_hat * k)

    version: Rr, restricted, heuristic approach
             Ru, uniform, random assignment
    '''
    n_dim, m_dim = X.shape

    A_t = np.zeros((k_cluster, k_cluster))
    B_t = np.zeros((m_dim, k_cluster))
    x_sum = 0
    alpha_sum = 0

    # step 1: sample n_hat * k_cluster points as initial X_hat.
    X_0 = np.zeros((m_dim, n_hat))
    for idx in range(n_hat):
        sample_idx = np.random.randint(0, n_dim)
        x_sample = X[sample_idx, :]
        X_0[:, idx] = x_sample


    # step 1: initialization, get X_hat (including clusters info)
    # and W_hat from X_0, using same init as in CNMF.
    # here representative_size_count is the n_1_hat, n_2_hat, ..., n_k_hat.
    t1 = time.time()
    X_hat, W_hat, representative_size_count = initialize_X_W_hat(X_0, k_cluster)
    X_0, W_0 = X_hat.copy(), W_hat.copy()
    t2 = time.time()
    # print('init cost {:.4f}'.format(t2 - t1))
    
    # step 2: after initialization of X_hat, update alpha, W_hat and X_hat alternatively.
    t_start = time.time()
    print(lmda, _NF, eps)
    g_hat_list = []
    error_eval_list = []
    t_cur = 0
    for t in range(T):
        # t_start_online = time.time()
        g_hat_i = get_g_hat_value(t, W_hat, X_hat, A_t, B_t,
                x_sum, alpha_sum)
        D_tmp_dummy = X_hat @ W_hat
        error_i = eval_g_hat_with_DnX(X, D_tmp_dummy.T, n_dim, m_dim)
        g_hat_list.append((t_cur, g_hat_i))
        error_eval_list.append((t_cur, error_i))
        if t % 50 == 0 and flag:
            D_t = np.matmul(X_hat, W_hat)
            tmp_assignment = get_clustering_assignment_1(X, D_t, k_cluster)
            tmp_acc, tmp_AMI = evaluation_clustering(tmp_assignment, y_true)
            print('1)iteration {}, distance acc = {:.4f}, AMI = {:.4f}'.format(t, tmp_acc, tmp_AMI))

            tmp_assignment = get_clustering_assignment_2(X, D_t, k_cluster, lmda)
            tmp_acc, tmp_AMI = evaluation_clustering(tmp_assignment, y_true)
            print('2)iteration {}, kmeans of weights acc = {:.4f}, AMI = {:.4f}'.format(t, tmp_acc, tmp_AMI))
            t_end = time.time()
            print('time elapse = {:.4f}s'.format(t_end - t_start))
            t_start = t_end

            print('-' * 7)


        sample_idx = np.random.randint(0, n_dim)
        x_sample = X[sample_idx, :]

        # update alpha
        t1 = time.time()
        lars_lasso = LassoLars(alpha = lmda, max_iter = 500)
        D_t = np.matmul(X_hat, W_hat)
        lars_lasso.fit(D_t, x_sample)
        alpha_t = lars_lasso.coef_
        # t2 = time.time()
        # print('lasso cost {:.4f}s'.format(t2 - t1))
        
        # using different clustering assignment
        # t1 = time.time()
        if version == 'Rr':
            cluster_of_x_i = np.argmax(alpha_t)
        # elif version == 'Ru':
        else:
            cluster_of_x_i = int(np.random.uniform(0, k_cluster))
        # t2 = time.time()
        # print('argmax alpha cost {:.4f}s'.format(t2 - t1))

        # t1 = time.time()
        A_t += np.matmul(alpha_t.reshape(k_cluster, 1), alpha_t.reshape(1, k_cluster))
        B_t += np.matmul(x_sample.reshape(m_dim, 1), alpha_t.reshape(1, k_cluster))
        x_sum += (np.linalg.norm(x_sample) ** 2)
        alpha_sum += lmda * np.linalg.norm(alpha_t, 1)
        # t2 = time.time()
        # print('update At, Bt cost {:.4f}s'.format(t2 - t1))


        # update X_hat
        # t1 = time.time()
        W_hat, X_hat = update_W_X_hat(W_hat, X_hat, representative_size_count, x_sample, cluster_of_x_i, 
                A_t, B_t, x_sum, alpha_sum, t, eps)
        t2 = time.time()
        t_cur += (t2 - t1)
        # print('update X_hat, W_hat cost {:.4f}s'.format(t2 - t1))

    print('Dcitionary update done! Time elapse {:.04f}s'.format(time.time() - t_start))

    return W_hat, X_hat, representative_size_count, X_0, W_0, g_hat_list, error_eval_list

def initialize_X_W_hat(X_0, k_cluster):
    '''
    takes intial collection of X and number of cluster as input,
    run k-Means on it, return the sorted (by cluster) X_hat, W_hat,
    and number of points in each cluster, i.e. n_hat_i
    '''
    # this function takes the initialziation step of CNMF and gives a X_hat, W_hat
    # cluster X_hat, get X_hat, W_0 as output of some method, and assignment of X_0
    # kmeans works with row vector, however, X_0 is a column vec matrix.
    kmeans = KMeans(n_clusters = k_cluster, max_iter = 1000)
    kmeans.fit(X_0.T)
    X_hat_assignments = kmeans.labels_


    # now we need to classify the X_hat to X_1, X_2, X_3
    # by using a dictionary candidate_clusters
    candidate_clusters = {x:np.array([]) for x in set(X_hat_assignments)}
    for idx, label in enumerate(X_hat_assignments):
        if candidate_clusters[label].size == 0:
            candidate_clusters[label] = X_0[:, idx]
        else:
            candidate_clusters[label] = np.vstack((candidate_clusters[label], X_0[:, idx]))

    X_hat = np.array([])
    check_list = []
    sorted_assignment = []
    for label in candidate_clusters:
        candidate_clusters[label] = candidate_clusters[label].T
        shape_of_cluster = candidate_clusters[label].shape
        print('label {} has shape of: {}'.format(label, shape_of_cluster))
        check_list.append(shape_of_cluster[1])
        if X_hat.size == 0:
            X_hat = candidate_clusters[label]
            sorted_assignment = [label] * shape_of_cluster[1]
        else:
            X_hat = np.hstack((X_hat, candidate_clusters[label]))
            sorted_assignment += [label] * shape_of_cluster[1]

    sorted_assignment = np.array(sorted_assignment)

    # based on the CNMF paper, we start the initialization with fresh k-Means
    # H: R^{n * k} matrix, indicate the cluster assignments
    # centroids can be calculated as F = X*W*D^{-1}, Where D: R^{k * k} is the count diagonal matrix
    # then we can say W = H*D^{-1}
    m_dim, n_dim = X_hat.shape
    cluster_count = [len(np.where(X_hat_assignments == i)[0]) for i in range(k_cluster)]
    assert cluster_count == check_list
    
    D = np.zeros((k_cluster, k_cluster), int)
    for idx in range(k_cluster):
        D[idx][idx] = cluster_count[idx] + 1e-3

    H = np.zeros((n_dim, k_cluster), int)
    for idx in range(k_cluster):
        non_zero_idx = np.where(sorted_assignment == idx)[0]
        H[non_zero_idx, idx] = 1
    
    W_hat = np.matmul((H + np.ones(H.shape, int) * 0.2), np.linalg.inv(D))

    return X_hat, W_hat, cluster_count


def update_W_X_hat(W_hat, X_hat, repre_size_count, x_sample, cluster_of_x_i, 
        A_t, B_t, x_sum, alpha_sum, t, eps):
    # add W_hat block diagonal constraint,
    # using projection.
    # linalg.init()

    # W_hat_gpu = gpuarray.to_gpu(W_hat.astype(np.float64))
    # tmp_x = np.ascontiguousarray(X_hat)
    # X_hat_gpu = gpuarray.to_gpu(tmp_x.astype(np.float64))
    # A_t_gpu = gpuarray.to_gpu(A_t.astype(np.float64))
    # B_t_gpu = gpuarray.to_gpu(B_t.astype(np.float64))


    cluster_seperation_idx = np.cumsum(repre_size_count)
    end_idx = cluster_seperation_idx[cluster_of_x_i]
    start_idx = end_idx - repre_size_count[cluster_of_x_i]
    A_t_inv = np.linalg.pinv(A_t)

    # W_opt_old_X = opt_cal_W_hat_numpy(W_hat, X_hat, A_t, B_t, x_sum, alpha_sum, eps, t)
    W_opt_old_X = opt_cal_W_hat_solve(W_hat, X_hat, A_t_inv, B_t, x_sum, alpha_sum, eps, t)
    g_hat_old_X = get_g_hat_value(t, W_opt_old_X, X_hat, A_t, B_t, x_sum, alpha_sum)

    # W_opt_old_X = update_W_hat_skcuda(W_hat_gpu, X_hat_gpu, A_t_gpu, B_t_gpu, 
    #       x_sum, alpha_sum, eps, t)
    # g_hat_old_X = get_g_hat_value(t, W_opt_old_X.get(), X_hat, A_t, B_t, x_sum, alpha_sum)

    list_of_W_opt_new_X = [W_opt_old_X]
    list_of_g_hat_new_X = [g_hat_old_X]
    list_of_new_X = [X_hat]

    # print('starting loop in update_W_X, total {}'.format(end_idx - start_idx))
    for idx in range(start_idx, end_idx):
        # print('iter # {}'.format(idx))
        t1 = time.time()
        X_hat_new  = X_hat.copy()
        X_hat_new[:, idx] = x_sample
        list_of_new_X.append(X_hat_new)
        # tmp_x = np.ascontiguousarray(X_hat_new)
        # X_hat_new_gpu = gpuarray.to_gpu(tmp_x.astype(np.float64))
        t2 = time.time()
        # print('\t update X_hat cost {:.4f}s'.format(t2 - t1))

        t1 = time.time()
        # W_opt_new_X = opt_cal_W_hat_numpy(W_hat, X_hat_new, A_t, B_t, x_sum, alpha_sum, eps, t)
        # W_opt_new_X = update_W_hat_numpy(W_hat, X_hat_new, A_t, B_t, x_sum, alpha_sum, eps, t)
        W_opt_new_X = opt_cal_W_hat_solve(W_hat, X_hat_new, A_t_inv, B_t, x_sum, alpha_sum, eps, t)
        g_hat_new_X = get_g_hat_value(t, W_opt_new_X, X_hat_new, A_t, B_t, x_sum, alpha_sum)

        # W_opt_new_X = update_W_hat_skcuda(W_hat_gpu, X_hat_new_gpu, A_t_gpu, B_t_gpu, 
        #       x_sum, alpha_sum, eps, t)
        # g_hat_new_X = get_g_hat_value(t, W_opt_new_X.get(), X_hat_new, A_t, B_t, x_sum, alpha_sum)
        t2 = time.time()
        # print('\t update W_hat_new cost {:.4f}'.format(t2 - t1))

        t1 = time.time()
        list_of_W_opt_new_X.append(W_opt_new_X)
        list_of_g_hat_new_X.append(g_hat_new_X)
        t2 = time.time()
        # print('appending W_opt list cost {:.4f}s'.format(t2 - t1))

    min_g_idx = np.argmin(list_of_g_hat_new_X)

    X_hat_new = list_of_new_X[min_g_idx]
    W_hat_new = list_of_W_opt_new_X[min_g_idx]
    # if list_of_g_hat_new_X[min_g_idx] <= g_hat_old_X:
    #     X_hat_new = X_hat.copy()
    #     X_hat_new[:, start_idx + min_g_idx] = x_sample
    #     # W_hat_new = list_of_W_opt_new_X[min_g_idx].get()
    #     W_hat_new = list_of_W_opt_new_X[min_g_idx].copy()
    # else:
    #     X_hat_new = X_hat.copy()
    #     # W_hat_new = W_opt_old_X.get()
    #     W_hat_new = W_opt_old_X.copy()


    return W_hat_new, X_hat_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numIter', type=int, default=1200)
    parser.add_argument('--lmda', type=float, default=1e-1)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--normal_factor', '--NF', type=float, default=200)
    parser.add_argument('--file_name', type=str, default='tmp_pic')
    parser.add_argument('--dtype', type=str, default='scRNA',
            choices=['scRNA', 'synthetic', 
                'synthetic_1', 'synthetic_2'],
            help='synthetic1: well sep, 2: close cluster')
    parser.add_argument('--k_cluster', type=int, default=10)
    parser.add_argument('--csize', type=int, default=5000,
            help='size of each cluster, integer, default 500')
    parser.add_argument('--candidate_size', type=int, default=15)
    parser.add_argument('--pca', type=int, default = 100)
    parser.add_argument('--numAver', type=int, default=1)
    args = parser.parse_args()

    # set number of iteration, lambda in lasso, epsilon in dictionary update and normalization factor
    print(args)
    numIter = args.numIter
    lmda = args.lmda
    eps = args.eps
    _NF = args.normal_factor
    file_name = args.file_name
    k_cluster = args.k_cluster
    # by experiment, the 10000 sample case has t_cmf > t_ocmf
    cluster_size = args.csize
    candidate_set_size = args.candidate_size
    P_component = args.pca
    aver_num = args.numAver

    data_type = args.dtype

    # np.random.seed(42)
    data_root = '/home/jianhao2/'
    data_root_shared = '/data/shared/jianhao/'
    # df_file = os.path.join(data_root, 'pandas_dataframe')
    # feat_file = os.path.join(data_root, 'df_feature_column')
    if data_type == 'synthetic':
        k_cluster = 10
        df_name = 'df_synthetic_well_sep'
        fc_name = 'feature_column_synthetic_well_sep'
        df_file = os.path.join(data_root_shared, '10xGenomics_scRNA/pandasDF', df_name) 
        feat_file = os.path.join(data_root_shared, '10xGenomics_scRNA/pandasDF', fc_name)
    elif data_type == 'synthetic_1':
        k_cluster = 10
        df_name = 'df_synthetic_disjoint_{}'.format(cluster_size)
        fc_name = 'feature_column_synthetic_disjoint_{}'.format(cluster_size)
        df_file = os.path.join(data_root_shared, 'synthetic_data', df_name) 
        feat_file = os.path.join(data_root_shared, 'synthetic_data', fc_name)
    elif data_type == 'synthetic_2':
        k_cluster = 10
        df_name = 'df_synthetic_overlap_{}'.format(cluster_size)
        fc_name = 'feature_column_synthetic_overlap_{}'.format(cluster_size)
        df_file = os.path.join(data_root_shared, 'synthetic_data', df_name) 
        feat_file = os.path.join(data_root_shared, 'synthetic_data', fc_name)
    elif data_type == 'scRNA':
        k_cluster = 10
        df_name = 'pandas_dataframe_10_clusters_-1'
        fc_name = 'df_feature_column_10_clusters_-1'
        df_file = os.path.join(data_root_shared, '10xGenomics_scRNA/pandasDF', df_name)
        feat_file = os.path.join(data_root_shared, '10xGenomics_scRNA/pandasDF', fc_name)

    # np.random.seed(42)
    df = pd.read_pickle(df_file)
    with open(feat_file, 'rb') as f:
        feat_cols = pickle.load(f)
    X_raw = df[feat_cols].values
    X_raw = X_raw - np.min(X_raw) + 0.1
    Y = df['label'].values



    # # ----------------------------------------------------
    # X_for_nmf = normalize(X_raw) * _NF
    # D_nmf, label_nmf = nmf_clustering(X_for_nmf, k_cluster, numIter = 1000)
    # acc_nmf, AMI_nmf = evaluation_clustering(label_nmf, Y)

    # print(' ------ final accuracy = {:.4f}, AMI = {:.4f}'.format(acc_nmf, AMI_nmf))


    # ----------------------------------------------------
    # use PCA to reduce X_raw to [num_of_cells * number of PCA componets]

    if P_component != -1:
        pca = PCA(n_components = P_component)
        # X_pca_all = pca.fit_transform(np.vstack((X_raw, D_nmf)))
        # X_pca = X_pca_all[:-k_cluster, :]

        X_pca = pca.fit_transform(X_raw)
    else:
        X_pca = X_raw
    pca_cols = ['Principle component {}'.format(i) for i in range(X_pca.shape[1])]

    X = my_normalize(X_pca) * _NF
    # X = X_pca

    n_dim, m_dim = X.shape

    # ----------------------------------------------------
    # 1) online cvxMF, our algorithm. to determine the lower bound of time.
    n_hat = k_cluster * candidate_set_size
    t_ocmf = 0
    acc = 0
    acc_array = []
    for round_num in range(aver_num):
        t1 = time.time()
        W_hat_tmp, X_hat_tmp, repre_size_count_tmp, X_0_tmp, W_0_tmp, g_hat_list_tmp, error_list_tmp = cvx_online_dict_learning(X, 
                Y, n_hat, k_cluster, 
                numIter, lmda, eps,
                flag = False, version = 'Ru')
        t2 = time.time()
        t_ocmf += (t2 - t1)
        D_final_tmp = np.matmul(X_hat_tmp, W_hat_tmp)

        # clustered_label = get_clustering_assignment_1(X, D_final)
        clustered_label_ocmf = get_clustering_assignment_2(X, D_final_tmp, k_cluster, lmda)
        acc_tmp, AMI_tmp = evaluation_clustering(clustered_label_ocmf, Y)
        acc_array.append(acc_tmp)
        if acc_tmp >= acc:
            W_hat = W_hat_tmp
            X_hat = X_hat_tmp
            X_0 = X_0_tmp
            W_0 = W_0_tmp
            D_final = D_final_tmp
            acc = acc_tmp
            AMI = AMI_tmp
            repre_size_count = repre_size_count_tmp
            g_hat_list_ocmf = g_hat_list_tmp
            error_list_ocmf = error_list_tmp
        if acc >= 0.9:
            break
    acc_aver = np.mean(acc_array)
    t_ocmf = t_ocmf / (round_num + 1)
    print(' ------ ocmf final accuracy = {:.4f}, AMI = {:.4f}'.format(acc, AMI))
    t_lower_bound = t_ocmf


    # ----------------------------------------------------
    # 2) Then, traditional NMF
    # D_nmf_pca = X_pca_all[-k_cluster:, :]
    if np.min(X_pca) < 0:
        X_for_nmf = X_pca - np.min(X_pca)
    else:
        X_for_nmf = X_pca
    # X_for_nmf = normalize(X_for_nmf) * _NF
    X_for_nmf = my_normalize(X_for_nmf) * _NF
    # D_nmf, _, label_nmf = nmf_clustering(X_for_nmf, k_cluster, numIter = 1000)
    # D_nmf_pca = pca.transform(D_nmf)
    # ipdb.set_trace()
    t1 = time.time()
    D_nmf, _, label_nmf, error_list_nmf = my_nmf_clustering(X_for_nmf, k_cluster, 
            numIter = numIter, t_lower_bound = t_lower_bound)
    t2 = time.time()
    t_nmf = t2 - t1
    acc_nmf, AMI_nmf = evaluation_clustering(label_nmf, Y)

    print(' ------ nmf final accuracy = {:.4f}, AMI = {:.4f}'.format(acc_nmf, AMI_nmf))

    # ----------------------------------------------------
    # 3)  compare with online NMF in their paper
    D_0 = (X_0 @ W_0).reshape(m_dim, k_cluster)
    # D_0 = normalize(D_0, axis = 0) * _NF

    acc_omf = 0
    AMI_omf = 0
    acc_omf_array = []
    t_omf = 0

    for round_num in range(aver_num):
        t1 = time.time()
        D_omf_final_tmp, error_list_omf = my_online_dict_learning(X, lmda = lmda, 
                D_0 = D_0, T = numIter, k_cluster = k_cluster, 
                eps = eps, _NF = _NF, t_lower_bound = t_lower_bound)
        t2 = time.time()
        t_omf += (t2 - t1)

        clustered_label_omf = get_clustering_assignment_2(X, D_omf_final_tmp, 
                k_cluster, lmda)
        acc_omf_tmp, AMI_omf_tmp = evaluation_clustering(clustered_label_omf, Y)
        acc_omf_array.append(acc_omf_tmp)
        if acc_omf_tmp >= acc_omf:
            D_omf_final = D_omf_final_tmp
            acc_omf, AMI_omf = acc_omf_tmp, AMI_omf_tmp
        if acc_omf >= 0.9:
            break
    acc_aver_omf = np.mean(acc_omf_array)
    t_omf = t_omf/(round_num + 1)

    print(' ------ onlineMF final accuracy = {:.4f}, AMI = {:.4f}'.format(acc_omf, 
        AMI_omf))

    
    # ----------------------------------------------------
    # 4) CNMF in jordan's paper.
    t1 = time.time()
    W_cnmf, _, clustered_label_cnmf, error_list_cmf = my_CNMF(X.T, 
            k_cluster, max_iter = numIter * 1, eps = eps,
            t_lower_bound = t_lower_bound)
    D_cnmf = (X.T @ W_cnmf).T
    t2 = time.time()
    t_cmf = t2 - t1

    clustered_label_cnmf = get_clustering_assignment_2(X, D_cnmf, 
            k_cluster, lmda)
    acc_cnmf, AMI_cnmf = evaluation_clustering(clustered_label_cnmf, Y)
    print(' ------ cnmf final accuracy = {:.4f}, AMI = {:.4f}'.format(acc_cnmf, 
        AMI_cnmf))
    # ----------------------------------------------------

    max_len = len(error_list_ocmf)
    error_list_nmf = list(takespread(error_list_nmf, max_len))
    error_list_omf = list(takespread(error_list_omf, max_len))
    error_list_cmf = list(takespread(error_list_cmf, max_len))

    print('-' * 7)
    save_path = 'error_time_curve'
    for m in ['ocmf', 'nmf', 'cmf', 'omf']:
        f = 'time_error_list_{}'.format(m)
        f = os.path.join(save_path, f)
        if m == 'ocmf':
            list_to_save = error_list_ocmf
        elif m == 'nmf':
            list_to_save = error_list_nmf
        elif m == 'omf':
            list_to_save = error_list_omf
        elif m == 'cmf':
            list_to_save = error_list_cmf
        with open(f, 'w') as f:
            for line in list_to_save:
                f.write(str(line[0]) + ',' + str(line[1]))
                f.write('\n')
    # print(len(error_list_ocmf), 
    #         ['{:.4f}'.format(x[0]) for x in error_list_ocmf[-3:]],
    #         ['{:.4f}'.format(x[1]) for x in error_list_ocmf[:3]],
    #         ['{:.4f}'.format(x[1]) for x in error_list_ocmf[-3:]])
    # print(len(error_list_nmf), 
    #         ['{:.4f}'.format(x[0]) for x in error_list_nmf[-3:]],
    #         ['{:.4f}'.format(x[1]) for x in error_list_nmf[:3]],
    #         ['{:.4f}'.format(x[1]) for x in error_list_nmf[-3:]])
    # print(len(error_list_omf), 
    #         ['{:.4f}'.format(x[0]) for x in error_list_omf[-3:]],
    #         ['{:.4f}'.format(x[1]) for x in error_list_omf[:3]],
    #         ['{:.4f}'.format(x[1]) for x in error_list_omf[-3:]])
    # print(len(error_list_cmf), 
    #         ['{:.4f}'.format(x[0]) for x in error_list_cmf[-3:]],
    #         ['{:.4f}'.format(x[1]) for x in error_list_cmf[:3]],
    #         ['{:.4f}'.format(x[1]) for x in error_list_cmf[-3:]])
