#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script only test the accuracy of calculating g_hat funciton.
"""

import numpy as np
from cvx_online_NMF import get_g_hat_value
import time

m_dim = 5000
n_dim = 200
k = 3
n_hat = 50

X = np.random.randn(m_dim, n_dim)
X_hat = np.random.randn(m_dim, n_hat)
W_hat = np.random.randn(n_hat, k)
D_hat = X_hat @ W_hat

alpha = np.random.randn(k, n_dim)
lmda = 0.5

x_sum, alpha_sum = 0, 0
A_t = np.zeros((k, k))
B_t = np.zeros((m_dim, k))

T = 7
g_val_culmulative = 0
for idx in range(T):
    x_sample = X[:, idx]
    alpha_i = alpha[:, idx]
    A_t += alpha_i.reshape(k, 1) @ alpha_i.reshape(1, k)
    B_t += x_sample.reshape(m_dim, 1) @ alpha_i.reshape(1, k)
    x_sum += (np.linalg.norm(x_sample) ** 2)
    alpha_sum += lmda * np.linalg.norm(alpha_i, 1)

    T1 = 1/2 * (np.linalg.norm(x_sample.reshape(m_dim, 1) - X_hat @ W_hat @ alpha_i.reshape(k, 1)) ** 2)
    T2 = lmda * np.linalg.norm(alpha_i, 1)
    g_val_culmulative += (T1 + T2)

# g_cal = get_g_hat_value(T, W_hat, X_hat, A_t, B_t, x_sum, alpha_sum)
# g_add = g_val_culmulative/T
# print(g_cal)
# print(g_add)
# print((g_add - g_cal) <= 1e-8)
# 
# from cvx_online_NMF import initialize_X_W_hat
# X_hat, W_hat, cluster_count = initialize_X_W_hat(X, 2)
# np.set_printoptions(3)
# print(X_hat)
# print(W_hat)
# print(cluster_count)

from cvx_online_NMF import update_W_hat, update_W_hat_2, update_D_hat
from unnecessary_funcs import dict_update

print('-' * 7)
t1 = time.time()
W_new_2 = dict_update(X_hat @ W_hat, A_t, B_t, 1e-3)
t2 = time.time()
print('BCD of D cost time {:.06f}s'.format(t2 - t1))

print('-' * 7)
t1 = time.time()
# W_new_2 = update_W_hat_2(W_hat, X_hat, A_t, B_t, x_sum, alpha_sum, 1e-3)
t2 = time.time()
print('BCD of W cost time {:.06f}s'.format(t2 - t1))

print('-' * 7)
t2 = time.time()
W_new = update_D_hat(T, W_hat, X_hat, A_t, B_t, x_sum, alpha_sum, 0.1)
t3 = time.time()
print('D cvx cost time {:.06f}s'.format(t3 - t2))

print('-' * 7)
t3 = time.time()
W_new = update_W_hat(T, W_hat, X_hat, A_t, B_t, x_sum, alpha_sum, 0.1)
t4 = time.time()
print('W cvx cost time {:.06f}s'.format(t4 - t3))
