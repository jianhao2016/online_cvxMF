#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script contains function such as:
    get_g_value
    cluster evaluation
    etc.
"""
import ipdb
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from functools import reduce
from sklearn.decomposition import NMF
from collections import Counter
from cluster_assignment_method import get_clustering_assignment_1

def nmf_clustering(X, k_cluster, numIter):
    '''
    X: n x m, where n is # of sample and n is sample dimension
        i.e. each row of X is a sample
    return:
    D: k x m, i.e. centroid in rows.
    '''
    nmf_model = NMF(n_components = k_cluster, solver = 'mu', max_iter = numIter, alpha = 0)

    weight_matrix = nmf_model.fit_transform(X)
    centroid_matrix = nmf_model.components_

    _, assignment = kmeans_clustering(weight_matrix, k_cluster, numIter)

    return centroid_matrix, weight_matrix, assignment

def my_nmf_clustering(X, k_cluster, numIter):
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

    for it in range(numIter):
        weight_matrix = w_old * ((X @ c_old.T)/ (w_old @ c_old @ c_old.T))
        w_old = weight_matrix
        centroid_matrix = c_old * ((w_old.T @ X) / (w_old.T @ w_old @ c_old))
        c_old = centroid_matrix

    # err = np.linalg.norm(X - weight_matrix @ centroid_matrix)
    # print('l2 error = {:.4f}'.format(err))

    _, assignment = kmeans_clustering(weight_matrix, k_cluster, numIter)
    return centroid_matrix, weight_matrix, assignment


def kmeans_clustering(X, k_cluster, numIter):
    kmeans = KMeans(n_clusters = k_cluster, max_iter = numIter)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    assignment = kmeans.labels_
    return centroids, assignment


def eval_g_hat_with_DnX(X, D, n_dim, m_dim):
    '''
    error = || X - D \Alpha ||^2
    use sklearn regression to find alpha.
    X is a row vector matrix. n_dim * m_dim
    D is a row vector matrix, k_dim * m_dim
    '''
    row_X, col_X = X.shape
    assert row_X == n_dim and col_X == m_dim
    l2_reg = LinearRegression()
    l2_reg.fit(D.T, X.T)
    alpha = l2_reg.coef_

    ave_error = 1/n_dim * np.linalg.norm(X - alpha @ D)
    return ave_error
    


def get_g_hat_value(t, W_hat, X_hat, A_t, B_t, x_sum, alpha_sum):
    T1 = 1/2 * x_sum
    T2 = np.trace(np.matmul(B_t, np.matmul(W_hat.T, X_hat.T)))
    T3 = 1/2 * np.trace(reduce(np.matmul, (X_hat, W_hat, A_t, W_hat.T, X_hat.T)))
    T4 = alpha_sum
    g_value = 1/(t + 1) * (T1 - T2 + T3 + T4)
    return g_value
    
def evaluation_clustering(pred_label, true_label):
    # both input should be 1d array
    all_predict_label = list(set(pred_label))
    clusters_label_distn = {i : [] for i in all_predict_label}
    for pl, tl in zip(pred_label, true_label):
        if len(clusters_label_distn[pl]) == 0:
            clusters_label_distn[pl] = np.array([tl])
        else:
            clusters_label_distn[pl] = np.concatenate((clusters_label_distn[pl], np.array([tl])))

    for pl in clusters_label_distn:
        uniq, idx = np.unique(clusters_label_distn[pl], return_inverse=True)
        most_frequent_label = uniq[np.argmax(np.bincount(idx))]
        clusters_label_distn[pl] = most_frequent_label

    pred_label_str = np.array(list(map(lambda x: clusters_label_distn[x], pred_label)))
    # print(pred_label_str)
    all_true_label_list = list(set(true_label))
    label_str_2_int = {}
    for idx in range(len(all_true_label_list)):
        label_str_2_int[all_true_label_list[idx]] = idx
    pred_label_int = np.array(list(map(lambda x: label_str_2_int[x], pred_label_str)))
    true_label_int = np.array(list(map(lambda x: label_str_2_int[x], true_label)))

    acc = accuracy_score(true_label_int, pred_label_int)
    AMI = adjusted_mutual_info_score(true_label_int, pred_label_int, average_method = 'arithmetic')
    return acc, AMI

def geo_projection_to_cvx_cmb(w_list, tol=1e-5):
    '''
    take input np.array 1d,
    project to the constraint set of 
    convex combination, i.e.
        w_i >= 0, sum(w) = 1
    w_proj[i] = max(w[i] + lambda1, 0), 
    sum(w_proj[i]) = 1
    use binary search to find lambda1
    '''
    num_w = len(w_list)
    l_upper = -min(w_list) + 1/num_w
    l_lower = -max(w_list)
    constraint = lambda x: np.sum(np.clip(w_list + x, 0, None))
    l_tmp = (l_upper + l_lower) / 2
    k = 0
    while abs(constraint(l_tmp) - 1) > tol:
        if constraint(l_tmp) > 1:
            l_upper = l_tmp
        else:
            l_lower = l_tmp
        k += 1
        if k >= 1000:
            print('over 1000 iteration in projection')
            break
        l_tmp = (l_upper + l_lower) / 2
    # print('iteration in projection: ', k)
    proj_w = np.clip(w_list + l_tmp, 0, None)
    return proj_w

def get_idx_list_from_cluster(cluster_size_count):
    num_cluster = len(cluster_size_count)
    total_list = np.arange(sum(cluster_size_count))
    idx_list = []
    for label in range(num_cluster):
        start_idx = sum(cluster_size_count[:label])
        end_idx = sum(cluster_size_count[:label + 1])
        idx_list.append(list(total_list[start_idx:end_idx]))
    return idx_list

def binsearch_min(fn, grad, upper, lower):
    # miu_upper = 1000
    # miu_lower = -1000
    miu_upper = upper
    miu_lower = lower
    miu_mid = (miu_lower + miu_upper) / 2
    k = 100
    while abs(miu_upper - miu_lower) > 1e-30 and k >= 0:
        if grad(miu_mid) * grad(miu_upper) >= 0:
            miu_upper = miu_mid
        elif grad(miu_mid) * grad(miu_lower) > 0:
            miu_lower = miu_mid
        miu_mid = (miu_upper + miu_lower) / 2
        k -= 1
        # print(fn(miu_mid))

    # print(fn(miu_upper))
    # print('-' * 7)
    return miu_upper

def proj_D_onto_X_hat(D_t_j_col, X_hat_j, X_hat_j_norm_inv):
    '''
    D_t_j_col is the jth column of D_t matrix,
    X_hat_j is the xs chosen for jth cluster in X_hat,
    X_hat_j_norm_inv = (X_hat_j.T @ X_hat_j)^-1
    D_j_new = sum_k(beta_k * x_k) for k in cluster j

    by solving the constrained opt problem,
    beta = {(X_hat_j.T @ X_hat_j)^-1 * [X_hat_j.T @ D_t_j_col + miu]}+
    use binary search for beta
    '''
    n_hat = X_hat_j.shape[1]
    vec_1 = X_hat_j_norm_inv @ X_hat_j.T @ D_t_j_col
    vec_2 = np.sum(X_hat_j_norm_inv, axis = 1)
    constraint = lambda miu: np.sum(np.clip(vec_1 + miu * vec_2, 0, None))
    grad_sum = lambda miu: np.sum((vec_1 + miu* vec_2 > 0) * vec_2)

    pos_vec2 = (vec_2 > 0) * vec_2
    pos_vec1 = (vec_2 > 0) * vec_1
    upper_miu = max(1, (1 - sum(pos_vec1))/sum(pos_vec2))

    neg_vec2 = (vec_2 < 0) * vec_2
    neg_vec1 = (vec_2 < 0) * vec_1
    lower_miu = min(-1, (1 - sum(neg_vec1))/sum(neg_vec2))

    miu_tmp = binsearch_min(constraint, grad_sum, upper_miu, lower_miu)
    if constraint(miu_tmp) > 1:
        print('value equals 1 does not exist')
        return None

    miu_right_lower, miu_left_upper = miu_tmp, miu_tmp
    # miu_right_upper = 1000
    # miu_left_lower = -1000
    miu_right_upper = upper_miu
    miu_left_lower = lower_miu
    miu_right_mid = (miu_right_upper + miu_right_lower) / 2
    miu_left_mid = (miu_left_upper + miu_left_lower) / 2

    k = 100
    while abs(miu_right_upper - miu_right_lower) > 1e-30 and k >= 0:
        if constraint(miu_right_mid) > 1:
            miu_right_upper = miu_right_mid
        else:
            miu_right_lower = miu_right_mid
        if constraint(miu_left_mid) > 1:
            miu_left_lower = miu_left_mid
        else:
            miu_left_upper = miu_left_mid
        k -= 1

        miu_left_mid = (miu_left_upper + miu_left_lower)/2
        miu_right_mid = (miu_right_upper + miu_right_lower)/2
        
        # print(constraint(miu_left_mid))
        # print(constraint(miu_right_mid))
        # print('---')

    # print(constraint(miu_left_mid))
    # print(constraint(miu_right_mid))
    if abs(constraint(miu_left_mid) - 1) < 1e-5:
        miu_mid = miu_left_mid
    elif abs(constraint(miu_right_mid) == 1) < 1e-5:
        miu_mid = miu_right_mid
    else:
        print('none of the center equal one!')
        miu_mid = 0
    beta = np.clip(vec_1 + miu_mid * vec_2, 0, None)
    return beta

def get_centroids_order_from_D_and_X(X_mat, D_mat, k_cluster):
    '''
    input: 
        X_mat: m * n, column represent a sample.
        D_mat: m * k_cluster, column represent centroids
    output:
        a order list
    the order of X_mat by col will be [type-1, type-2, ..., type-k]
    but D_mat is not in same order, [cluster-1, cluster-2, ..., cluster-k]
    we need to resorted D in a list [a-1, a-2, ..., a-k]
    such that 
        cluster-i => a-i => type-j, 
    is the right assignment. so 
        idx, type in enumerate(a_list):
    can find the centroids as well as the right color
    '''
    m_dim, n_dim = X_mat.shape
    num_sample_per_cluster = n_dim // k_cluster

    centroid_color_list = [k_cluster] * k_cluster
    D_norm_mat = []
    norm_of_dist_from_cent = lambda X, d: np.linalg.norm(X - d.reshape(-1, 1))
    for col_idx in range(k_cluster):
        D_k = D_mat[:, col_idx]
        D_dist_list = []
        for label in range(k_cluster):
            X_hat_start_idx = label * num_sample_per_cluster
            X_hat_end_idx = (label + 1) * num_sample_per_cluster
            X_hat_k_cluster = X_mat[:, X_hat_start_idx:X_hat_end_idx]
            D_dist_tmp = norm_of_dist_from_cent(X_hat_k_cluster, D_k)
            D_dist_list.append(D_dist_tmp)
        D_norm_mat.append(D_dist_list)
        # D_k_color_idx = np.argmin(D_dist_list)
        # centroid_color_list.append(D_k_color_idx)
    D_norm_mat = np.array(D_norm_mat)
    max_val = np.max(D_norm_mat) + 42

    for _ in range(k_cluster):
        cur_min = np.argmin(D_norm_mat)
        D_idx, cluster_idx = np.unravel_index(cur_min, D_norm_mat.shape)
        centroid_color_list[D_idx] = cluster_idx
        D_norm_mat[D_idx, :] = max_val
        D_norm_mat[:, cluster_idx] = max_val

    centroid_order_list = centroid_color_list
    return centroid_order_list

def get_centroids_order_from_assignment(assignment, k_cluster):
    '''
    input: 
        X_mat: m * n, column represent a sample.
        D_mat: m * k_cluster, column represent centroids
    output:
        a order list
    the order of X_mat by col will be [type-1, type-2, ..., type-k]
    but D_mat is not in same order, [cluster-1, cluster-2, ..., cluster-k]
    we need to resorted D in a list [a-1, a-2, ..., a-k]
    such that 
        cluster-i => a-i => type-j, 
    is the right assignment. so 
        idx, type in enumerate(a_list):
    can find the centroids as well as the right color
    '''
    assigned_label = assignment
    num_sample_per_cluster = len(assigned_label) // k_cluster
    centroid_order_list = []
    for label in range(k_cluster):
        sub_label_list = assigned_label[label * num_sample_per_cluster:(label + 1) * num_sample_per_cluster]
        c_tmp = Counter(list(sub_label_list))
        idx_of_centroid, _ = c_tmp.most_common()[0]
        centroid_order_list.append(idx_of_centroid)
    return centroid_order_list

def save_dict(d, path_2_file):
    print('dumpping dictionary to pickle file', path_2_file)
    with open(path_2_file, 'wb') as f:
        pickle.dump(d, f)
        print('Done dumpping!')

def load_dict(path_2_file):
    print('Loading dictionary from: ', path_2_file)
    with open(path_2_file, 'rb') as f:
        tmp = pickle.load(f)
        print('Done loading!')
    return tmp

if __name__ == '__main__':
    n_hat = 2
    m = 2
    X_hat_j = np.random.randn(m, n_hat) * 1
    X_hat_j_norm_inv = np.linalg.inv(X_hat_j.T @ X_hat_j)
    D_t_j_col = np.random.randn(m) * 1

    beta = proj_D_onto_X_hat(D_t_j_col, X_hat_j, X_hat_j_norm_inv)
    print('X = ')
    print(repr(X_hat_j))
    print('D_j = \n', repr(D_t_j_col))
    print('beta =', beta)
    print('sum = ', sum(beta))
    D_proj = X_hat_j @ beta.reshape((-1, 1))
    print('projected D = \n', repr(D_proj))


