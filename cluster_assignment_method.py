#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script keeps the clustering assignment method for evaluation.
Given data, and centroids (or equivalent), return assignments.
"""


import numpy as np
from sklearn.linear_model import LassoLars
from sklearn.cluster import KMeans

def get_clustering_assignment_1(X, D_centroids, k_cluster):
    n_dim, m_dim = X.shape
    centrioid_mat = np.reshape(D_centroids, (m_dim, k_cluster))

    dist_to_centroids = lambda x: np.array([np.linalg.norm(x - centrioid_mat[:, j]) for j in range(k_cluster)])
    assigned_cluster = np.argmin(list(map(dist_to_centroids, X.reshape(n_dim, -1))), axis  = 1)
    assignment = assigned_cluster

    return assignment

def get_clustering_assignment_2(X, D_centroids, k_cluster, lmda, numIter = 1000):
    n_dim, m_dim = X.shape
    centrioid_mat = np.reshape(D_centroids, (m_dim, k_cluster))
    weight_mat = np.zeros((n_dim, k_cluster))
    for idx in range(n_dim):
        lars_lasso = LassoLars(alpha = 0, max_iter = 500)
        lars_lasso.fit(centrioid_mat, X[idx, :])
        alpha_t = lars_lasso.coef_

        weight_mat[idx, :] = alpha_t

    kmeans = KMeans(n_clusters = k_cluster, max_iter = numIter)
    kmeans.fit(weight_mat)
    assignment = kmeans.labels_

    return assignment
    
