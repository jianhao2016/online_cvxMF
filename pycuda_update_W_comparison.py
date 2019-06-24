#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script contains a cuda version of update W_hat
"""

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
import skcuda.misc as misc
import time
import ipdb
from functools import reduce
# from cvx_online_NMF import update_W_hat
from common_functions import get_g_hat_value, geo_projection_to_cvx_cmb
from common_functions import proj_D_onto_X_hat

def update_W_hat_skcuda(W_hat, X_hat, A_t, B_t, x_sum, alpha_sum, eps, t):
    n_hat, k_cluster = W_hat.shape
    # m_dim, _ = X_hat.shape
    W_hat_new = W_hat.copy()
    linalg.init()

    if not isinstance(W_hat_new, gpuarray.GPUArray):
        W_hat_new_gpu = gpuarray.to_gpu(W_hat_new.astype(np.float64))
    else:
        W_hat_new_gpu = W_hat_new

    if not isinstance(X_hat, gpuarray.GPUArray):
        tmp_x = np.ascontiguousarray(X_hat)
        X_hat_gpu = gpuarray.to_gpu(tmp_x.astype(np.float64))
    else:
        X_hat_gpu = X_hat
    # X_hat_T_gpu = gpuarray.to_gpu(X_hat.T.copy().astype(np.float64))
    X_hat_T_gpu = linalg.transpose(X_hat_gpu)

    if not isinstance(A_t, gpuarray.GPUArray):
        A_t_gpu = gpuarray.to_gpu(A_t.astype(np.float64))
    else:
        A_t_gpu = A_t
    A_t_gpu_trans = linalg.transpose(A_t_gpu)

    if not isinstance(B_t, gpuarray.GPUArray):
        B_t_gpu = gpuarray.to_gpu(B_t.astype(np.float64))
    else:
        B_t_gpu = B_t
    B_t_gpu_trans = linalg.transpose(B_t_gpu)

    all_ones_gpu = gpuarray.to_gpu(np.ones((n_hat, 1), dtype=np.float64))

    k = 0
    while True:
        k += 1
        # ipdb.set_trace()
        W_hat_old_gpu = W_hat_new_gpu.copy()
        for j in range(k_cluster):
            T1 = linalg.dot(X_hat_T_gpu, B_t_gpu_trans[j, :].reshape((-1, 1)))
            X_product_gpu = linalg.dot(X_hat_T_gpu, X_hat_gpu)
            T2 = reduce(linalg.dot, (X_product_gpu,  W_hat_new_gpu, 
                A_t_gpu_trans[j, :].reshape(-1, 1)))
            grad_gpu = - T1 + T2
            step_size = 1/(linalg.norm(X_product_gpu) * linalg.norm(A_t_gpu_trans[j, :]) + 1e-8)
            tmp = - step_size * grad_gpu.reshape((-1)) + W_hat_new_gpu[:, j].copy()

            # u_j_gpu = 1/2 * (tmp + abs(tmp))
            # normalized_u_j_gpu = 1/max(linalg.norm(u_j_gpu), 1) * u_j_gpu

            # u_j_gpu = 1/max(linalg.norm(tmp), 1) * tmp
            # normalized_u_j_gpu = 1/2 * (u_j_gpu + abs(u_j_gpu))
            u_j = geo_projection_to_cvx_cmb(tmp.get())
            normalized_u_j_gpu = gpuarray.to_gpu(u_j.astype(np.float64))
            
            W_hat_new_gpu[:, j] = normalized_u_j_gpu

        # T1 = linalg.dot(X_hat_T_gpu, B_t_gpu)
        # X_product_gpu = linalg.dot(X_hat_T_gpu, X_hat_gpu)
        # T2 = reduce(linalg.dot, (X_product_gpu, W_hat_new_gpu, A_t_gpu))
        # grad_gpu =  T2 - T1
        # step_size = 1/(linalg.norm(X_product_gpu) * linalg.norm(A_t_gpu) + 1e-8)
        # tmp =  W_hat_new_gpu - step_size * grad_gpu
        # u_gpu = 1/2 * (tmp + abs(tmp))

        # column_sum_gpu = misc.sum(u_gpu, axis = 0).astype(np.float64)
        # # ipdb.set_trace()
        # div_mat_gpu = linalg.dot(all_ones_gpu, column_sum_gpu.reshape((1, -1))) + 1e-8
        # W_hat_new_gpu = u_gpu / div_mat_gpu.astype(np.float64)
        
        # if k % 50 == 0:
        #     g_val = get_g_hat_value(t, W_hat_new_gpu.get(), X_hat,
        #             A_t, B_t, x_sum, alpha_sum)
        #     print('iteration {}, function value: {:.4f}'.format(k, g_val))

        if (linalg.norm(W_hat_new_gpu - W_hat_old_gpu) < eps) or k >= 10000:
            break

    return W_hat_new_gpu

def update_W_hat_numpy(W_hat, X_hat, A_t, B_t, x_sum, alpha_sum, eps, t):
    n_hat, k_cluster = W_hat.shape
    W_hat_new = W_hat.copy()

    k = 0
    while True:
        k += 1
        W_hat_old = W_hat_new.copy()
        for _ in range(1):
            for j in range(k_cluster):
                T1 = np.matmul(X_hat.T, B_t[:, j])
                X_product = np.matmul(X_hat.T, X_hat)
                tmp = np.matmul(X_product, W_hat_new)
                T2 = np.matmul(tmp, A_t[:, j])
                grad = (T1 - T2)
                step_size = 1/(np.linalg.norm(X_product) * np.linalg.norm(A_t[:, j]) + 1e-8)
                u_j = step_size * grad + W_hat_new[:,j]
                # W_hat_new[:, j] = (u_j / max(np.linalg.norm(u_j), 1)) * 1
                # # project to non-zero
                # W_hat_new[:, j] = np.clip(W_hat_new[:, j], 0, None)
                W_hat_new[:, j] = geo_projection_to_cvx_cmb(u_j)

        # if k % 50 == 0:
        #     g_val = get_g_hat_value(t, W_hat_new, X_hat, A_t, B_t, x_sum, alpha_sum)
        #     print('iteration {}, function value: {:.4f}'.format(k, g_val))

        if (np.linalg.norm(W_hat_new - W_hat_old) < eps) or k >= 50000:
            break
    # print('g_val = {:.06f}'.format(g_val))

    return W_hat_new

def opt_cal_W_hat_numpy(W_hat, X_hat, A_t, B_t, x_sum, alpha_sum, eps, t):
    '''
    calculate optimal W_hat by setting gradient to zero, 
    and use pseudo inverse to get closed form
    grad_w = ( - X_hat.T * B_t + X_hat.T * X_hat * W * A_t) == 0
        ==> W_hat_opt = (X_hat.T * X_hat)^-1 * X_hat.T * B_t * (A_t)^-1
    '''
    X_hat_norm = X_hat.T @ X_hat
    T1 = np.linalg.pinv(X_hat_norm)
    T2 = X_hat.T @ B_t
    T3 = np.linalg.pinv(A_t)
    W_opt = T1 @ T2 @ T3
    for col_idx in range(W_opt.shape[1]):
        old_col = W_opt[:, col_idx]
        proj_col = geo_projection_to_cvx_cmb(old_col)
        W_opt[:, col_idx] = proj_col
    return W_opt

def opt_cal_W_hat_solve(W_hat, X_hat, A_t_inv, B_t, x_sum, alpha_sum, eps, t):
    '''
    calculate optimal W_hat by setting gradient to zero, 
    then use np.linalg.solve to solve the linear equation,
    b = ax
    '''
    b_mat = X_hat.T @ B_t @ A_t_inv
    # W_opt = np.linalg.solve(X_hat.T @ X_hat, b_mat)
    W_opt = np.linalg.lstsq(X_hat.T @ X_hat, b_mat, rcond = None)[0]
    for col_idx in range(W_opt.shape[1]):
        old_col = W_opt[:, col_idx]
        proj_col = geo_projection_to_cvx_cmb(old_col)
        W_opt[:, col_idx] = proj_col
    return W_opt

def update_D_and_project(D_t, A_t, B_t, eps, X_hat_j, X_hat_j_norm_inv):
    '''
    D_t: R^(m * k)
    A_t: R^(k * k)
    B_t: R^(m * k)
    '''
    m_dim, k_cluster = D_t.shape
    m_dim, n_hat = X_hat_j.shape
    D_new = D_t.copy()

    # t_start = time.time()
    W_hat_new = np.zeros((n_hat, k_cluster))
    while True:
        D_old = D_new.copy()
        for j in range(k_cluster):
            grad = (B_t[:, j] - np.matmul(D_new, A_t[:, j]))
            u_j =  1/(A_t[j, j] + 1e-5) * grad + D_new[:, j]
            beta = proj_D_onto_X_hat(u_j, X_hat_j, X_hat_j_norm_inv)
            W_hat_new[:, j] = beta
            D_new[:, j] = X_hat_j.dot(beta)
            # D_new[:, j] = (u_j / max(np.linalg.norm(u_j), 1)) * 200
        if (np.linalg.norm(D_new - D_old) < eps):
            break
    # print('Iteration in dictionary update cost {:.04f}s'.format(time.time() - t_start))

    return W_hat_new

if __name__ == '__main__':
    
    m_dim = 320
    n_dim = 200
    k = 10
    n_hat = 50
    
    X = np.random.randn(m_dim, n_dim)
    X_hat = np.random.randn(m_dim, n_hat)
    W_hat = np.random.randn(n_hat, k)
    W_hat_2 = W_hat.copy()
    D_hat = X_hat @ W_hat
    
    alpha = np.random.randn(k, n_dim)
    lmda = 0.5
    eps = 1e-5
    
    x_sum, alpha_sum = 0, 0
    A_t = np.zeros((k, k))
    B_t = np.zeros((m_dim, k))
    
    T = 70
    # g_val_culmulative = 0
    for tmp in range(T):
        idx = tmp % n_dim
        x_sample = X[:, idx]
        alpha_i = alpha[:, idx]
        A_t += alpha_i.reshape(k, 1) @ alpha_i.reshape(1, k)
        B_t += x_sample.reshape(m_dim, 1) @ alpha_i.reshape(1, k)
        x_sum += (np.linalg.norm(x_sample) ** 2)
        alpha_sum += lmda * np.linalg.norm(alpha_i, 1)
    
        # T1 = 1/2 * (np.linalg.norm(x_sample.reshape(m_dim, 1) - 
        #     X_hat @ W_hat @ alpha_i.reshape(k, 1)) ** 2)
        # T2 = lmda * np.linalg.norm(alpha_i, 1)
        # g_val_culmulative += (T1 + T2)

    t1 = time.time()
    for _ in range(10):
        W_cuda = update_W_hat_skcuda(W_hat, X_hat, A_t, B_t, x_sum, 
            alpha_sum, eps, T)
    t2 = time.time()
    print('pycuda cost {:.4f}s'.format(t2 - t1))
    # print('total number of loop {}, each cost {:.4f}s'.format(k, (t2-t1)/k))
    g_hat_val_cuda = get_g_hat_value(T, W_cuda.get(), X_hat, A_t, B_t, x_sum, alpha_sum)
    print('g value = {:.4f}'.format(g_hat_val_cuda))

    print('-' * 7)

    t1 = time.time()
    for _ in range(10):
        W_numpy = update_W_hat_numpy(W_hat, X_hat, A_t, B_t, x_sum, 
                alpha_sum, eps, T)
    t2 = time.time()
    print('numpy dense cost {:.4f}s'.format(t2 - t1))
    # print('total number of loop {}, each cost {:.4f}s'.format(k, (t2-t1)/k))
    g_hat_val_np = get_g_hat_value(T, W_numpy, X_hat, A_t, B_t, x_sum, alpha_sum)
    print('g value = {:.4f}'.format(g_hat_val_np))
    print('-' * 7)

    t1 = time.time()
    k = 1
    for _ in range(10):
        W_opt = opt_cal_W_hat_numpy(W_hat_2, X_hat, A_t, B_t, x_sum, 
                alpha_sum, eps, T)
    t2 = time.time()
    print('opt cal cost {:.4f}s'.format(t2 - t1))
    # print('total number of loop {}, each cost {:.4f}s'.format(k, (t2-t1)/k))
    g_hat_val_opt = get_g_hat_value(T, W_opt, X_hat, A_t, B_t, x_sum, alpha_sum)
    print('g value = {:.4f}'.format(g_hat_val_opt))
    print('-' * 7)

    A_inv = np.linalg.pinv(A_t)
    t1 = time.time()
    k = 1
    for _ in range(10):
        W_opt = opt_cal_W_hat_solve(W_hat_2, X_hat, A_inv, B_t, x_sum, 
                alpha_sum, eps, T)
    t2 = time.time()
    print('opt lsqst cost {:.4f}s'.format(t2 - t1))
    # print('total number of loop {}, each cost {:.4f}s'.format(k, (t2-t1)/k))
    g_hat_val_opt = get_g_hat_value(T, W_opt, X_hat, A_t, B_t, x_sum, alpha_sum)
    print('g value = {:.4f}'.format(g_hat_val_opt))
    print('-' * 7)

    # W_diff = W_cuda.get() - W_numpy
    # W_diff_l1_norm = np.linalg.norm(W_diff, 1)
    # print('l1 norm of W difference(cuda - numpy): {:.4f}'.format(W_diff_l1_norm))

    W_diff = W_numpy - W_opt
    W_diff_l1_norm = np.linalg.norm(W_diff, 1)
    print('l1 norm of W difference(numpy - opt): {:.4f}'.format(W_diff_l1_norm))
    
    t1 = time.time()
    k = 1
    X_hat_j = X_hat
    X_hat_j_norm_inv = np.linalg.pinv(X_hat.T @ X_hat)
    D_t = X_hat_j @ W_hat
    for _ in range(10):
        W_opt = update_D_and_project(D_t, A_t, B_t, eps, X_hat_j, X_hat_j_norm_inv)
    t2 = time.time()
    print('opt cal cost {:.4f}s'.format(t2 - t1))
    # print('total number of loop {}, each cost {:.4f}s'.format(k, (t2-t1)/k))
    g_hat_val_opt = get_g_hat_value(T, W_opt, X_hat, A_t, B_t, x_sum, alpha_sum)
    print('g value = {:.4f}'.format(g_hat_val_opt))
    print('-' * 7)

