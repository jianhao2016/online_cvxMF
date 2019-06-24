#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script contains udpate function using cvxpy for:
    updating D, in online NMF paper
    updating W_hat, in cvx_online_NMF.py
"""

import numpy as np
import cvxpy as cvx

def update_D_hat_cvxpy(t, W_hat, X_hat, A_t, B_t, x_sum, alpha_sum, eps):
    '''
    use cvxpy to update D
    '''
    n_hat, k_cluster = W_hat.shape
    D_hat = (X_hat @ W_hat)
    
    D = cvx.Variable(shape = D_hat.shape)
    # constraint = [W >= 0, W <= 1]
    constraint = [D >= 0, D <= 1]
    # T1 = cvx.trace(B_t * W.T * X_hat.T)
    # tmp = X_hat * W * A_t * W.T * X_hat.T
    # T2 = cvx.trace(X_hat * W * A_t * W.T * X_hat.T)
    # T2 = cvx.trace(W.T * W)
    XW = D
    T1 = cvx.sum(cvx.multiply(B_t, XW))

    m_dim = XW.shape[0]
    print('m_dim  = ', m_dim)
    quad_sum = 0
    for idx in range(m_dim):
        quad_sum += cvx.quad_form(XW[idx, :].T, A_t)

    T2 = quad_sum
    # tmp = XW * (A_t.T)
    # T2 = cvx.sum(cvx.multiply(XW, tmp))

    print('is T1 cvx? ', T1.is_convex())
    print('is T2 cvx? ', T2.is_convex())
    # print('tmp shape:', tmp.shape)
    print('T2 shape:', T2.shape)

    obj = cvx.Minimize(1/t * (1/2 * x_sum - T1 + 1/2 * T2 + alpha_sum))
    # obj = cvx.Minimize((1/t) * (1/2 * x_sum - cvx.trace(B_t * W.T * X_hat.T)
    #     + 1/2 * cvx.trace(X_hat * W * A_t * W.T * X_hat.T) + alpha_sum))
    prob = cvx.Problem(obj, constraint)
    # prob.solve(solver = cvx.CVXOPT)
    prob.solve(solver = cvx.OSQP)

    # if prob.status != cvx.OPTIMAL:
    #     raise Exception('CVX solver did not converge!')
    print('residual norm = {:.06f}'.format(prob.value))

    D_hat_new = D.value
    
    return D_hat_new


def update_W_hat_cvxpy(t, W_hat, X_hat, A_t, B_t, x_sum, alpha_sum, eps):
    '''
    use cvxpy to update W_hat
    '''
    n_hat, k_cluster = W_hat.shape
    W_hat_new = W_hat.copy()
    
    W = cvx.Variable(shape = W_hat.shape)
    # constraint = [W >= 0, W <= 1]
    constraint = [W >= 0]
    # T1 = cvx.trace(B_t * W.T * X_hat.T)
    # tmp = X_hat * W * A_t * W.T * X_hat.T
    # T2 = cvx.trace(X_hat * W * A_t * W.T * X_hat.T)
    # T2 = cvx.trace(W.T * W)
    XW = X_hat * W
    T1 = cvx.sum(cvx.multiply(B_t, XW))

    m_dim = XW.shape[0]
    print('m_dim  = ', m_dim)
    quad_sum = 0
    for idx in range(m_dim):
        quad_sum += cvx.quad_form(XW[idx, :].T, A_t)

    T2 = quad_sum
    # tmp = XW * (A_t.T)
    # T2 = cvx.sum(cvx.multiply(XW, tmp))

    print('is T1 cvx? ', T1.is_convex())
    print('is T2 cvx? ', T2.is_convex())
    # print('tmp shape:', tmp.shape)
    print('T2 shape:', T2.shape)

    obj = cvx.Minimize(1/t * (1/2 * x_sum - T1 + 1/2 * T2 + alpha_sum))
    # obj = cvx.Minimize((1/t) * (1/2 * x_sum - cvx.trace(B_t * W.T * X_hat.T)
    #     + 1/2 * cvx.trace(X_hat * W * A_t * W.T * X_hat.T) + alpha_sum))
    prob = cvx.Problem(obj, constraint)
    # prob.solve(solver = cvx.CVXOPT)
    prob.solve(solver = cvx.OSQP)

    # if prob.status != cvx.OPTIMAL:
    #     raise Exception('CVX solver did not converge!')
    print('residual norm = {:.06f}'.format(prob.value))

    
    W_hat_new = W.value

    g_val = get_g_hat_value(1, W_hat_new, X_hat, A_t, B_t, x_sum, alpha_sum)
    print('g_val = {:.06f}'.format(g_val))
    return W_hat_new

