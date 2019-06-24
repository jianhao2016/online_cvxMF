#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script contain visualization for both 
    scRNA/synthetic dataset
    image dataset
"""

import os
import sys
import time
from pylab import *

# the following code is just to add other modules
# i.e. cluster_assignment_method, common_function, etc.
parent_path = os.path.abspath('../')
if parent_path not in sys.path:
    sys.path.append(parent_path)

import matplotlib.pyplot as plt
import numpy as np
from common_functions import get_centroids_order_from_D_and_X, get_centroids_order_from_assignment
from sklearn.manifold import TSNE

def get_cmap(n, name='gist_rainbow'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
    color_map = plt.cm.get_cmap(name, n)
    rgb_list = [color_map(idx) for idx in range(n)]
    return rgb_list

def plot_diff_method(df_final, pca_cols, n_dim, k_cluster, accuracy_dict, 
        representative_size_count, size_of_cluster, 
        cluster_size_count = None):
    ###### ------ the following code is for visualization!!!
    # run tSNE for visualization
    # rndperm = np.random.permutation(df_final.shape[0])
    # n_sne = k_cluster * 3 + n_dim + n_hat

    acc_nmf, AMI_nmf = accuracy_dict['nmf']
    acc_cnmf, AMI_cnmf = accuracy_dict['cnmf']
    acc_noncvx, AMI_noncvx = accuracy_dict['onmf']
    acc, AMI = accuracy_dict['ocnmf']

    t_start = time.time()
    if len(pca_cols) > 2:
        tsne = TSNE(n_components = 2, verbose = 1, perplexity=40, n_iter = 1000, random_state = 42)
        tsne_result = tsne.fit_transform(df_final[pca_cols].values)
    elif len(pca_cols) == 2:
        tsne_result = df_final[pca_cols].values
    print('tsne_result shape = ', tsne_result.shape)

    print('t-SNE done! Time elapse {:.04f}s'.format(time.time() - t_start))
    df_tsne = df_final.copy()
    df_tsne['x-tsne'] = tsne_result[:, 0]
    df_tsne['y-tsne'] = tsne_result[:, 1]

    sc_x = df_tsne['x-tsne'].values[:n_dim]
    sc_y = df_tsne['y-tsne'].values[:n_dim]
    sc_types = df_tsne['label'].values[:n_dim]

    X_hat_x = df_tsne['x-tsne'].values[n_dim:-k_cluster * 4]
    X_hat_y = df_tsne['y-tsne'].values[n_dim:-k_cluster * 4]
    X_hat_types = df_tsne['label'].values[n_dim:-k_cluster * 4]

    cnmf_centroid_x = df_tsne['x-tsne'].values[-k_cluster * 4: -k_cluster * 3]
    cnmf_centroid_y = df_tsne['y-tsne'].values[-k_cluster * 4: -k_cluster * 3]
    cnmf_centroid_types = df_tsne['label'].values[-k_cluster * 4 : -k_cluster * 3]

    nmf_centroid_x = df_tsne['x-tsne'].values[-k_cluster * 3: -k_cluster * 2]
    nmf_centroid_y = df_tsne['y-tsne'].values[-k_cluster * 3: -k_cluster * 2]
    nmf_centroid_types = df_tsne['label'].values[-k_cluster * 3 : -k_cluster * 2]

    centroid_x = df_tsne['x-tsne'].values[-k_cluster * 2: -k_cluster]
    centroid_y = df_tsne['y-tsne'].values[-k_cluster * 2: -k_cluster]
    centroid_types = df_tsne['label'].values[-k_cluster * 2 : -k_cluster]

    non_cvx_centroid_x = df_tsne['x-tsne'].values[-k_cluster:]
    non_cvx_centroid_y = df_tsne['y-tsne'].values[-k_cluster:]
    non_cvx_centroid_types = df_tsne['label'].values[-k_cluster:]

    #### the code below are for color matching between samples and centroids
    #### an naive case will be let order_list = list(range(k_cluster))
    # nmf_centroid_order_list = list(range(k_cluster))
    # ocnmf_centroid_order_list = list(range(k_cluster))
    # onmf_centroid_order_list = list(range(k_cluster))
    X_mat_sc = np.hstack((sc_x.reshape(-1, 1), sc_y.reshape(-1, 1)))
    X_mat_sc = X_mat_sc.T

    D_mat_nmf = np.hstack((nmf_centroid_x.reshape(-1, 1),
        nmf_centroid_y.reshape(-1, 1))).T
    D_mat_cnmf = np.hstack((cnmf_centroid_x.reshape(-1, 1),
        cnmf_centroid_y.reshape(-1, 1))).T
    D_mat_ocnmf = np.hstack((centroid_x.reshape(-1, 1),
        centroid_y.reshape(-1, 1))).T
    D_mat_onmf = np.hstack((non_cvx_centroid_x.reshape(-1, 1), 
        non_cvx_centroid_y.reshape(-1, 1))).T

    # ipdb.set_trace()
    nmf_centroid_order_list = get_centroids_order_from_D_and_X(X_mat_sc, 
            D_mat_nmf, 
            k_cluster)

    cnmf_centroid_order_list = get_centroids_order_from_D_and_X(X_mat_sc, 
            D_mat_cnmf, 
            k_cluster)

    ocnmf_centroid_order_list = get_centroids_order_from_D_and_X(X_mat_sc, 
            D_mat_ocnmf, 
            k_cluster)
    
    onmf_centroid_order_list = get_centroids_order_from_D_and_X(X_mat_sc, 
            D_mat_onmf, 
            k_cluster)

    
    #### start ploting
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    color_list = get_cmap(k_cluster)
    marker_list = ['D'] * k_cluster
    centroid_color_list = color_list

    X_hat_index = np.zeros(1 + k_cluster)
    X_hat_index[1:] = np.cumsum(representative_size_count)

    if cluster_size_count == None:
        # not specify how many samples in each cluster, use uniform set
        cluster_size_count = [size_of_cluster] * k_cluster
    sample_index = np.zeros(1 + k_cluster)
    sample_index[1:] = np.cumsum(cluster_size_count)

    # plot data points only
    horizontal_margin = 0.1 * (max(sc_x) -  min(sc_x))
    vertical_margin = 0.1 * (max(sc_y) - min(sc_y))
    ax1 = fig.add_subplot(321)
    # ax1 = subplot2grid((3,4), (0, 0), colspan=2)
    # tmp = size_of_cluster
    for idx in range(k_cluster):
        x_start_idx = int(sample_index[idx])
        x_end_idx = int(sample_index[idx + 1])
        ax1.scatter(sc_x[x_start_idx:x_end_idx],
                sc_y[x_start_idx:x_end_idx],
                color = color_list[idx],
                label = list(set(sc_types))[idx], s = 50, alpha = 0.6)

        # ax1.scatter(sc_x[idx*tmp:(idx+1)*tmp], sc_y[idx*tmp:(idx+1)*tmp], 
        #         color = color_list[idx],
        #         # color = 'grey',
        #         label = list(set(sc_types))[idx], s = 50, alpha = 0.6)
    # ax1.legend(loc='upper left', prop={'size': 4})
    
    tmp_handles, tmp_labels = ax1.get_legend_handles_labels()
    order_index = sorted(range(len(tmp_labels)), key=tmp_labels.__getitem__)
    handles = [tmp_handles[idx] for idx in order_index]
    labels = [tmp_labels[idx] for idx in order_index]
    ax1.set_xlim(left = min(sc_x) - horizontal_margin, 
            right = max(sc_x) + horizontal_margin)
    ax1.set_ylim(bottom = min(sc_y) - vertical_margin, 
            top = max(sc_y) + vertical_margin)
    ax1.set_title('Original data')
    ax1.set_ylabel('tSNE element 2')
    ax1.set_xticklabels([])

    # ax_legend = subplot2grid((3, 4), (0, 2), colspan=2)
    ax_legend = fig.add_subplot(322)
    ax_legend.set_axis_off()
    legend = ax_legend.legend(handles, labels, 
            prop = {'size':7},
            loc='center', ncol = 2)


    # plot the result of ordinary nmf
    ax2 = fig.add_subplot(323)
    ax2.scatter(sc_x, sc_y, color = 'grey')
    # for idx in range(k_cluster):
    # ipdb.set_trace()
    for centroid_idx, color_idx in enumerate(nmf_centroid_order_list):
        ax2.scatter(nmf_centroid_x[centroid_idx], nmf_centroid_y[centroid_idx], 
                color = centroid_color_list[color_idx], 
                marker = marker_list[color_idx],
                alpha = 0.9, label = nmf_centroid_types[centroid_idx])
    # ax2.set_title('{}\nAccuracy: {:.4f}, AMI: {:.4f}'.format('NMF', acc_nmf, AMI_nmf))
    ax2.set_xlim(left = min(sc_x) - horizontal_margin, 
            right = max(sc_x) + horizontal_margin)
    ax2.set_ylim(bottom = min(sc_y) - vertical_margin, 
            top = max(sc_y) + vertical_margin)
    ax2.set_title('{}'.format('MF with bases'))
    # ax2.legend(loc = 'upper left', prop={'size': 4})
    # ax2.set_xlim(left = (min(sc_x) - (max(sc_x) - min(sc_x))/1))
    ax2.set_ylabel('tSNE element 2')
    ax2.set_xticklabels([])
    # ax2.set_yticklabels([])


    # plot the result of convex nmf
    ax_cvx = fig.add_subplot(324)
    ax_cvx.scatter(sc_x, sc_y, color = 'grey')
    # for idx in range(k_cluster):
    # ipdb.set_trace()
    for centroid_idx, color_idx in enumerate(cnmf_centroid_order_list):
        ax_cvx.scatter(cnmf_centroid_x[centroid_idx], cnmf_centroid_y[centroid_idx], 
                color = centroid_color_list[color_idx], 
                marker = marker_list[color_idx],
                alpha = 0.9, label = nmf_centroid_types[centroid_idx])
    # ax_cvx.set_title('{}\nAccuracy: {:.4f}, AMI: {:.4f}'.format('CNMF', acc_cnmf, AMI_cnmf))
    ax_cvx.set_xlim(left = min(sc_x) - horizontal_margin, 
            right = max(sc_x) + horizontal_margin)
    ax_cvx.set_ylim(bottom = min(sc_y) - vertical_margin, 
            top = max(sc_y) + vertical_margin)
    ax_cvx.set_title('{}'.format('cvxMF with bases'))
    # ax_cvx.legend(loc = 'upper left', prop={'size': 4})
    # ax_cvx.set_xlim(left = (min(sc_x) - (max(sc_x) - min(sc_x))/1))
    ax_cvx.set_xticklabels([])
    ax_cvx.set_yticklabels([])


    # plot the online NMF
    ax3 = fig.add_subplot(325)
    ax3.scatter(sc_x, sc_y, color = 'grey')
    # for idx in range(k_cluster):
    for centroid_idx, color_idx in enumerate(onmf_centroid_order_list):
        ax3.scatter(non_cvx_centroid_x[centroid_idx], non_cvx_centroid_y[centroid_idx], 
                color = centroid_color_list[color_idx], 
                marker = marker_list[color_idx],
                alpha = 0.9, label = non_cvx_centroid_types[centroid_idx])
    # ax3.set_title('{}\nAccuracy: {:.4f}, AMI: {:.4f}'.format('online_NMF', acc_noncvx, AMI_noncvx))
    ax3.set_xlim(left = min(sc_x) - horizontal_margin, 
            right = max(sc_x) + horizontal_margin)
    ax3.set_ylim(bottom = min(sc_y) - vertical_margin, 
            top = max(sc_y) + vertical_margin)
    ax3.set_title('{}'.format('online MF with bases'))
    ax3.set_xlabel('tSNE element 1')
    ax3.set_ylabel('tSNE element 2')
    # ax3.set_xlim(left = (min(sc_x) - (max(sc_x) - min(sc_x))/1))
    # ax3.legend(loc = 'upper left', prop={'size': 4})
    

    # plot cvx online NMF
    ax4 = fig.add_subplot(326)
    ax4.scatter(sc_x, sc_y, color = 'grey')
    # for idx in range(k_cluster):
    for centroid_idx, color_idx in enumerate(ocnmf_centroid_order_list):
        x_hat_start_idx = int(X_hat_index[centroid_idx])
        x_hat_end_idx = int(X_hat_index[centroid_idx + 1])
        ax4.scatter(X_hat_x[x_hat_start_idx:x_hat_end_idx], X_hat_y[x_hat_start_idx:x_hat_end_idx], 
                facecolor = color_list[color_idx], 
                # label = list(set(X_hat_types))[idx], 
                marker = 'x', s = 50, alpha = 0.3)
    

    for centroid_idx, color_idx in enumerate(ocnmf_centroid_order_list):
        ax4.scatter(centroid_x[centroid_idx], centroid_y[centroid_idx], 
                color = centroid_color_list[color_idx], 
                marker = marker_list[color_idx],
                alpha = 0.9, label = centroid_types[centroid_idx])
    
    # lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # fig.suptitle('tSNE of {} on scRNA expression\nAccuracy: {:.4f}, AMI: {:.4f}'.format('online_cvx_NMF', acc, AMI))
    # ax4.set_title('{}\nAccuracy: {:.4f}, AMI: {:.4f}'.format('online_cvx_NMF', acc, AMI))
    ax4.set_xlim(left = min(sc_x) - horizontal_margin, 
            right = max(sc_x) + horizontal_margin)
    ax4.set_ylim(bottom = min(sc_y) - vertical_margin, 
            top = max(sc_y) + vertical_margin)
    ax4.set_title('online cvxMF with bases and representative set')
    ax4.set_xlabel('tSNE element 1')
    ax4.set_yticklabels([])
    # ax4.set_xlim(left = (min(sc_x) - (max(sc_x) - min(sc_x))/1))
    # ax4.legend(loc = 'upper left', prop={'size': 4})


    ax1.grid(color = 'grey', alpha = 0.4)
    ax2.grid(color = 'grey', alpha = 0.4)
    ax_cvx.grid(color = 'grey', alpha = 0.4)
    ax3.grid(color = 'grey', alpha = 0.4)
    ax4.grid(color = 'grey', alpha = 0.4)
    # plt.tight_layout()

    return fig

def plot_diff_method_new(df_final, pca_cols, n_dim, k_cluster, accuracy_dict, 
        representative_size_count_Rr, representative_size_count_Ru,
        size_of_cluster, cluster_size_count = None):
    ###### ------ the following code is for visualization!!!
    # run tSNE for visualization
    # rndperm = np.random.permutation(df_final.shape[0])
    # n_sne = k_cluster * 3 + n_dim + n_hat

    acc_nmf, AMI_nmf = accuracy_dict['nmf']
    acc_cmf, AMI_cmf = accuracy_dict['cmf']
    acc_omf, AMI_omf = accuracy_dict['omf']
    acc_ocmf_Rr, AMI_ocmf_Rr = accuracy_dict['ocmf_rr']
    acc_ocmf_Ru, AMI_ocmf_Ru = accuracy_dict['ocmf_ru']

    t_start = time.time()
    if len(pca_cols) > 2:
        tsne = TSNE(n_components = 2, verbose = 1, perplexity=40, n_iter = 1000, random_state = 42)
        tsne_result = tsne.fit_transform(df_final[pca_cols].values)
    elif len(pca_cols) == 2:
        tsne_result = df_final[pca_cols].values
    print('tsne_result shape = ', tsne_result.shape)

    print('t-SNE done! Time elapse {:.04f}s'.format(time.time() - t_start))
    df_tsne = df_final.copy()
    df_tsne['x-tsne'] = tsne_result[:, 0]
    df_tsne['y-tsne'] = tsne_result[:, 1]

    sc_x = df_tsne['x-tsne'].values[:n_dim]
    sc_y = df_tsne['y-tsne'].values[:n_dim]
    sc_types = df_tsne['label'].values[:n_dim]

    Rr_candidate_size = np.sum(representative_size_count_Rr)
    Rr_end_idx = n_dim + Rr_candidate_size
    X_hat_Rr_x = df_tsne['x-tsne'].values[n_dim:Rr_end_idx]
    X_hat_Rr_y = df_tsne['y-tsne'].values[n_dim:Rr_end_idx]
    X_hat_Rr_types = df_tsne['label'].values[n_dim:Rr_end_idx]

    Ru_candidate_size = np.sum(representative_size_count_Rr)
    Ru_end_idx = Rr_end_idx + Ru_candidate_size
    X_hat_Ru_x = df_tsne['x-tsne'].values[Rr_end_idx:Ru_end_idx]
    X_hat_Ru_y = df_tsne['y-tsne'].values[Rr_end_idx:Ru_end_idx]
    X_hat_Ru_types = df_tsne['label'].values[Rr_end_idx:Ru_end_idx]

    centroid_Ru_x = df_tsne['x-tsne'].values[-k_cluster * 5: -k_cluster * 4]
    centroid_Ru_y = df_tsne['y-tsne'].values[-k_cluster * 5: -k_cluster * 4]
    centroid_Ru_types = df_tsne['label'].values[-k_cluster * 5 : -k_cluster * 4]

    cnmf_centroid_x = df_tsne['x-tsne'].values[-k_cluster * 4: -k_cluster * 3]
    cnmf_centroid_y = df_tsne['y-tsne'].values[-k_cluster * 4: -k_cluster * 3]
    cnmf_centroid_types = df_tsne['label'].values[-k_cluster * 4 : -k_cluster * 3]

    nmf_centroid_x = df_tsne['x-tsne'].values[-k_cluster * 3: -k_cluster * 2]
    nmf_centroid_y = df_tsne['y-tsne'].values[-k_cluster * 3: -k_cluster * 2]
    nmf_centroid_types = df_tsne['label'].values[-k_cluster * 3 : -k_cluster * 2]

    centroid_Rr_x = df_tsne['x-tsne'].values[-k_cluster * 2: -k_cluster]
    centroid_Rr_y = df_tsne['y-tsne'].values[-k_cluster * 2: -k_cluster]
    centroid_Rr_types = df_tsne['label'].values[-k_cluster * 2 : -k_cluster]

    non_cvx_centroid_x = df_tsne['x-tsne'].values[-k_cluster:]
    non_cvx_centroid_y = df_tsne['y-tsne'].values[-k_cluster:]
    non_cvx_centroid_types = df_tsne['label'].values[-k_cluster:]

    #### the code below are for color matching between samples and centroids
    #### an naive case will be let order_list = list(range(k_cluster))
    # nmf_centroid_order_list = list(range(k_cluster))
    # ocnmf_centroid_order_list = list(range(k_cluster))
    # onmf_centroid_order_list = list(range(k_cluster))
    X_mat_sc = np.hstack((sc_x.reshape(-1, 1), sc_y.reshape(-1, 1)))
    X_mat_sc = X_mat_sc.T

    D_mat_nmf = np.hstack((nmf_centroid_x.reshape(-1, 1),
        nmf_centroid_y.reshape(-1, 1))).T
    D_mat_cnmf = np.hstack((cnmf_centroid_x.reshape(-1, 1),
        cnmf_centroid_y.reshape(-1, 1))).T
    D_mat_ocnmf_Rr = np.hstack((centroid_Rr_x.reshape(-1, 1),
        centroid_Rr_y.reshape(-1, 1))).T
    D_mat_ocnmf_Ru = np.hstack((centroid_Ru_x.reshape(-1, 1),
        centroid_Ru_y.reshape(-1, 1))).T
    D_mat_onmf = np.hstack((non_cvx_centroid_x.reshape(-1, 1), 
        non_cvx_centroid_y.reshape(-1, 1))).T

    # ipdb.set_trace()
    nmf_centroid_order_list = get_centroids_order_from_D_and_X(X_mat_sc, 
            D_mat_nmf, 
            k_cluster)

    cnmf_centroid_order_list = get_centroids_order_from_D_and_X(X_mat_sc, 
            D_mat_cnmf, 
            k_cluster)

    ocnmf_Rr_centroid_order_list = get_centroids_order_from_D_and_X(X_mat_sc, 
            D_mat_ocnmf_Rr, 
            k_cluster)
    
    ocnmf_Ru_centroid_order_list = get_centroids_order_from_D_and_X(X_mat_sc, 
            D_mat_ocnmf_Ru, 
            k_cluster)
    
    onmf_centroid_order_list = get_centroids_order_from_D_and_X(X_mat_sc, 
            D_mat_onmf, 
            k_cluster)

    
    #### start ploting
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.55, wspace=0.1)
    color_list = get_cmap(k_cluster)
    marker_list = ['D'] * k_cluster
    centroid_color_list = color_list

    X_hat_index_Rr = np.zeros(1 + k_cluster)
    X_hat_index_Rr[1:] = np.cumsum(representative_size_count_Rr)

    X_hat_index_Ru = np.zeros(1 + k_cluster)
    X_hat_index_Ru[1:] = np.cumsum(representative_size_count_Ru)

    if cluster_size_count == None:
        # not specify how many samples in each cluster, use uniform set
        cluster_size_count = [size_of_cluster] * k_cluster
    sample_index = np.zeros(1 + k_cluster)
    sample_index[1:] = np.cumsum(cluster_size_count)

    # plot data points only
    horizontal_margin = 0.1 * (max(sc_x) -  min(sc_x))
    vertical_margin = 0.1 * (max(sc_y) - min(sc_y))
    ax1 = fig.add_subplot(321)
    # ax1 = subplot2grid((3,4), (0, 0), colspan=2)
    # tmp = size_of_cluster
    for idx in range(k_cluster):
        x_start_idx = int(sample_index[idx])
        x_end_idx = int(sample_index[idx + 1])
        ax1.scatter(sc_x[x_start_idx:x_end_idx],
                sc_y[x_start_idx:x_end_idx],
                color = color_list[idx],
                label = list(set(sc_types))[idx], s = 50, alpha = 0.6)

        # ax1.scatter(sc_x[idx*tmp:(idx+1)*tmp], sc_y[idx*tmp:(idx+1)*tmp], 
        #         color = color_list[idx],
        #         # color = 'grey',
        #         label = list(set(sc_types))[idx], s = 50, alpha = 0.6)
    # ax1.legend(loc='upper left', prop={'size': 4})
    
    tmp_handles, tmp_labels = ax1.get_legend_handles_labels()
    order_index = sorted(range(len(tmp_labels)), key=tmp_labels.__getitem__)
    handles = [tmp_handles[idx] for idx in order_index]
    labels = [tmp_labels[idx] for idx in order_index]
    ax1.set_xlim(left = min(sc_x) - horizontal_margin, 
            right = max(sc_x) + horizontal_margin)
    ax1.set_ylim(bottom = min(sc_y) - vertical_margin, 
            top = max(sc_y) + vertical_margin)
    ax1.set_title('Original data')
    ax1.set_ylabel('tSNE element 2')
    ax1.set_xticklabels([])


    # plot the result of ordinary nmf
    ax2 = fig.add_subplot(322)
    ax2.scatter(sc_x, sc_y, color = 'grey')
    # for idx in range(k_cluster):
    # ipdb.set_trace()
    for centroid_idx, color_idx in enumerate(nmf_centroid_order_list):
        ax2.scatter(nmf_centroid_x[centroid_idx], nmf_centroid_y[centroid_idx], 
                color = centroid_color_list[color_idx], 
                marker = marker_list[color_idx],
                alpha = 0.9, label = nmf_centroid_types[centroid_idx])
    # ax2.set_title('{}\nAccuracy: {:.4f}, AMI: {:.4f}'.format('NMF', acc_nmf, AMI_nmf))
    ax2.set_xlim(left = min(sc_x) - horizontal_margin, 
            right = max(sc_x) + horizontal_margin)
    ax2.set_ylim(bottom = min(sc_y) - vertical_margin, 
            top = max(sc_y) + vertical_margin)
    ax2.set_title('MF \n accuracy:{:.4f}'.format(acc_nmf))
    # ax2.legend(loc = 'upper left', prop={'size': 4})
    # ax2.set_xlim(left = (min(sc_x) - (max(sc_x) - min(sc_x))/1))
    # ax2.set_ylabel('tSNE element 2')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])


    # plot the result of convex nmf
    ax3 = fig.add_subplot(323)
    ax3.scatter(sc_x, sc_y, color = 'grey')
    # for idx in range(k_cluster):
    # ipdb.set_trace()
    for centroid_idx, color_idx in enumerate(cnmf_centroid_order_list):
        ax3.scatter(cnmf_centroid_x[centroid_idx], cnmf_centroid_y[centroid_idx], 
                color = centroid_color_list[color_idx], 
                marker = marker_list[color_idx],
                alpha = 0.9, label = nmf_centroid_types[centroid_idx])
    # ax_cvx.set_title('{}\nAccuracy: {:.4f}, AMI: {:.4f}'.format('CNMF', acc_cnmf, AMI_cnmf))
    ax3.set_xlim(left = min(sc_x) - horizontal_margin, 
            right = max(sc_x) + horizontal_margin)
    ax3.set_ylim(bottom = min(sc_y) - vertical_margin, 
            top = max(sc_y) + vertical_margin)
    ax3.set_title('cvxMF \n accuracy:{:.4f}'.format(acc_cmf))
    # ax_cvx.legend(loc = 'upper left', prop={'size': 4})
    # ax_cvx.set_xlim(left = (min(sc_x) - (max(sc_x) - min(sc_x))/1))
    ax3.set_xticklabels([])
    ax3.set_ylabel('tSNE element 2')
    # ax_cvx.set_yticklabels([])


    # plot the online NMF
    ax4 = fig.add_subplot(324)
    ax4.scatter(sc_x, sc_y, color = 'grey')
    # for idx in range(k_cluster):
    for centroid_idx, color_idx in enumerate(onmf_centroid_order_list):
        ax4.scatter(non_cvx_centroid_x[centroid_idx], non_cvx_centroid_y[centroid_idx], 
                color = centroid_color_list[color_idx], 
                marker = marker_list[color_idx],
                alpha = 0.9, label = non_cvx_centroid_types[centroid_idx])
    # ax3.set_title('{}\nAccuracy: {:.4f}, AMI: {:.4f}'.format('online_NMF', acc_noncvx, AMI_noncvx))
    ax4.set_xlim(left = min(sc_x) - horizontal_margin, 
            right = max(sc_x) + horizontal_margin)
    ax4.set_ylim(bottom = min(sc_y) - vertical_margin, 
            top = max(sc_y) + vertical_margin)
    ax4.set_title('online MF \n accuracy:{:.4f}'.format(acc_omf))
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    # ax_cvx.set_yticklabels([])
    # ax4.set_xlabel('tSNE element 1')
    # ax4.set_ylabel('tSNE element 2')
    # ax3.set_xlim(left = (min(sc_x) - (max(sc_x) - min(sc_x))/1))
    # ax3.legend(loc = 'upper left', prop={'size': 4})
    

    # plot cvx online NMF, Rr
    ax5 = fig.add_subplot(325)
    ax5.scatter(sc_x, sc_y, color = 'grey')
    # for idx in range(k_cluster):
    for centroid_idx, color_idx in enumerate(ocnmf_Rr_centroid_order_list):
        x_hat_start_idx = int(X_hat_index_Rr[centroid_idx])
        x_hat_end_idx = int(X_hat_index_Rr[centroid_idx + 1])
        ax5.scatter(X_hat_Rr_x[x_hat_start_idx:x_hat_end_idx], X_hat_Rr_y[x_hat_start_idx:x_hat_end_idx], 
                facecolor = color_list[color_idx], 
                # label = list(set(X_hat_types))[idx], 
                marker = 'x', s = 50, alpha = 0.3)
    

    for centroid_idx, color_idx in enumerate(ocnmf_Rr_centroid_order_list):
        ax5.scatter(centroid_Rr_x[centroid_idx], centroid_Rr_y[centroid_idx], 
                color = centroid_color_list[color_idx], 
                marker = marker_list[color_idx],
                alpha = 0.9, label = centroid_Rr_types[centroid_idx])
    
    # lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # fig.suptitle('tSNE of {} on scRNA expression\nAccuracy: {:.4f}, AMI: {:.4f}'.format('online_cvx_NMF', acc, AMI))
    # ax4.set_title('{}\nAccuracy: {:.4f}, AMI: {:.4f}'.format('online_cvx_NMF', acc, AMI))
    ax5.set_xlim(left = min(sc_x) - horizontal_margin, 
            right = max(sc_x) + horizontal_margin)
    ax5.set_ylim(bottom = min(sc_y) - vertical_margin, 
            top = max(sc_y) + vertical_margin)
    # ax5.set_title('online cvxMF $R_r$ with \n bases and representative sets')
    ax5.set_title('online cvxMF ($R_r$) \n accuracy:{:.4f}'.format(acc_ocmf_Rr))
    ax5.set_xlabel('tSNE element 1')
    ax5.set_ylabel('tSNE element 2')
    # ax4.set_xlim(left = (min(sc_x) - (max(sc_x) - min(sc_x))/1))
    # ax4.legend(loc = 'upper left', prop={'size': 4})

    # plot cvx online NMF, Ru
    ax6 = fig.add_subplot(326)
    ax6.scatter(sc_x, sc_y, color = 'grey')
    # for idx in range(k_cluster):
    for centroid_idx, color_idx in enumerate(ocnmf_Ru_centroid_order_list):
        x_hat_start_idx = int(X_hat_index_Ru[centroid_idx])
        x_hat_end_idx = int(X_hat_index_Ru[centroid_idx + 1])
        ax6.scatter(X_hat_Ru_x[x_hat_start_idx:x_hat_end_idx], X_hat_Ru_y[x_hat_start_idx:x_hat_end_idx], 
                facecolor = color_list[color_idx], 
                # label = list(set(X_hat_types))[idx], 
                marker = 'x', s = 50, alpha = 0.3)
    

    for centroid_idx, color_idx in enumerate(ocnmf_Ru_centroid_order_list):
        ax6.scatter(centroid_Ru_x[centroid_idx], centroid_Ru_y[centroid_idx], 
                color = centroid_color_list[color_idx], 
                marker = marker_list[color_idx],
                alpha = 0.9, label = centroid_Ru_types[centroid_idx])
    
    ax6.set_xlim(left = min(sc_x) - horizontal_margin, 
            right = max(sc_x) + horizontal_margin)
    ax6.set_ylim(bottom = min(sc_y) - vertical_margin, 
            top = max(sc_y) + vertical_margin)
    # ax6.set_title('online cvxMF $R_u$ with \n bases and representative sets')
    ax6.set_title('online cvxMF ($R_u$) \n accuracy:{:.4f}'.format(acc_ocmf_Ru))
    ax6.set_yticklabels([])
    ax6.set_xlabel('tSNE element 1')
    # ax4.set_xlim(left = (min(sc_x) - (max(sc_x) - min(sc_x))/1))
    # ax4.legend(loc = 'upper left', prop={'size': 4})


    ax1.grid(color = 'grey', alpha = 0.4)
    ax2.grid(color = 'grey', alpha = 0.4)
    ax3.grid(color = 'grey', alpha = 0.4)
    ax4.grid(color = 'grey', alpha = 0.4)
    ax5.grid(color = 'grey', alpha = 0.4)
    ax6.grid(color = 'grey', alpha = 0.4)
    # plt.tight_layout()

    return fig

def plot_diff_method_scRNA(df_final, pca_cols, n_dim, k_cluster, accuracy_dict, 
        representative_size_count_Rr, representative_size_count_Ru, 
        size_of_cluster, cluster_size_count = None):
    ###### ------ the following code is for visualization!!!
    # run tSNE for visualization
    # rndperm = np.random.permutation(df_final.shape[0])
    # n_sne = k_cluster * 3 + n_dim + n_hat

    acc_omf, AMI_omf = accuracy_dict['omf']
    acc_Ru, AMI_Ru = accuracy_dict['ocnmf_Ru']
    acc_Rr, AMI_Rr = accuracy_dict['ocnmf_Rr']


    t_start = time.time()
    if len(pca_cols) > 2:
        tsne = TSNE(n_components = 2, verbose = 1, perplexity=40, n_iter = 1000, random_state = 42)
        tsne_result = tsne.fit_transform(df_final[pca_cols].values)
    elif len(pca_cols) == 2:
        tsne_result = df_final[pca_cols].values
    print('tsne_result shape = ', tsne_result.shape)

    print('t-SNE done! Time elapse {:.04f}s'.format(time.time() - t_start))
    df_tsne = df_final.copy()
    df_tsne['x-tsne'] = tsne_result[:, 0]
    df_tsne['y-tsne'] = tsne_result[:, 1]

    sc_x = df_tsne['x-tsne'].values[:n_dim]
    sc_y = df_tsne['y-tsne'].values[:n_dim]
    sc_types = df_tsne['label'].values[:n_dim]

    Rr_candidate_size = np.sum(representative_size_count_Rr)
    Rr_end_idx = n_dim + Rr_candidate_size
    X_hat_Rr_x = df_tsne['x-tsne'].values[n_dim:Rr_end_idx]
    X_hat_Rr_y = df_tsne['y-tsne'].values[n_dim:Rr_end_idx]
    X_hat_Rr_types = df_tsne['label'].values[n_dim:Rr_end_idx]

    Ru_candidate_size = np.sum(representative_size_count_Rr)
    Ru_end_idx = Rr_end_idx + Ru_candidate_size
    X_hat_Ru_x = df_tsne['x-tsne'].values[Rr_end_idx:Ru_end_idx]
    X_hat_Ru_y = df_tsne['y-tsne'].values[Rr_end_idx:Ru_end_idx]
    X_hat_Ru_types = df_tsne['label'].values[Rr_end_idx:Ru_end_idx]

    centroid_x = df_tsne['x-tsne'].values[-k_cluster * 3: -k_cluster * 2]
    centroid_y = df_tsne['y-tsne'].values[-k_cluster * 3: -k_cluster * 2]
    centroid_types = df_tsne['label'].values[-k_cluster * 3 : -k_cluster * 2]

    centroid_Ru_x = df_tsne['x-tsne'].values[-k_cluster * 2: -k_cluster]
    centroid_Ru_y = df_tsne['y-tsne'].values[-k_cluster * 2: -k_cluster]
    centroid_Ru_types = df_tsne['label'].values[-k_cluster * 2 : -k_cluster]

    non_cvx_centroid_x = df_tsne['x-tsne'].values[-k_cluster:]
    non_cvx_centroid_y = df_tsne['y-tsne'].values[-k_cluster:]
    non_cvx_centroid_types = df_tsne['label'].values[-k_cluster:]

    #### the code below are for color matching between samples and centroids
    #### an naive case will be let order_list = list(range(k_cluster))
    # nmf_centroid_order_list = list(range(k_cluster))
    # ocnmf_centroid_order_list = list(range(k_cluster))
    # onmf_centroid_order_list = list(range(k_cluster))
    X_mat_sc = np.hstack((sc_x.reshape(-1, 1), sc_y.reshape(-1, 1)))
    X_mat_sc = X_mat_sc.T

    D_mat_ocnmf = np.hstack((centroid_x.reshape(-1, 1),
        centroid_y.reshape(-1, 1))).T
    D_mat_ocnmf_Ru = np.hstack((centroid_Ru_x.reshape(-1, 1),
        centroid_Ru_y.reshape(-1, 1))).T
    D_mat_onmf = np.hstack((non_cvx_centroid_x.reshape(-1, 1), 
        non_cvx_centroid_y.reshape(-1, 1))).T

    # ipdb.set_trace()
    ocnmf_centroid_order_list = get_centroids_order_from_D_and_X(X_mat_sc, 
            D_mat_ocnmf, 
            k_cluster)
    
    ocnmf_Ru_centroid_order_list = get_centroids_order_from_D_and_X(X_mat_sc, 
            D_mat_ocnmf_Ru, 
            k_cluster)
    
    onmf_centroid_order_list = get_centroids_order_from_D_and_X(X_mat_sc, 
            D_mat_onmf, 
            k_cluster)

    
    #### start ploting
    fig = plt.figure(figsize=(3.5, 1))
    plt.subplots_adjust(hspace=0.5, wspace=0.2)
    color_list = get_cmap(k_cluster)
    marker_list = ['D'] * k_cluster
    centroid_color_list = color_list

    X_hat_index = np.zeros(1 + k_cluster)
    X_hat_index[1:] = np.cumsum(representative_size_count_Rr)

    X_hat_index_Ru = np.zeros(1 + k_cluster)
    X_hat_index_Ru[1:] = np.cumsum(representative_size_count_Ru)


    if cluster_size_count == None:
        # not specify how many samples in each cluster, use uniform set
        cluster_size_count = [size_of_cluster] * k_cluster
    sample_index = np.zeros(1 + k_cluster)
    sample_index[1:] = np.cumsum(cluster_size_count)

    # plot data points only
    horizontal_margin = 0.1 * (max(sc_x) -  min(sc_x))
    vertical_margin = 0.1 * (max(sc_y) - min(sc_y))
    ax1 = fig.add_subplot(131)
    # ax1 = subplot2grid((3,4), (0, 0), colspan=2)
    # tmp = size_of_cluster
    for idx in range(k_cluster):
        x_start_idx = int(sample_index[idx])
        x_end_idx = int(sample_index[idx + 1])
        ax1.scatter(sc_x[x_start_idx:x_end_idx],
                sc_y[x_start_idx:x_end_idx],
                color = color_list[idx],
                label = list(set(sc_types))[idx], s = 50, alpha = 0.6)

        # ax1.scatter(sc_x[idx*tmp:(idx+1)*tmp], sc_y[idx*tmp:(idx+1)*tmp], 
        #         color = color_list[idx],
        #         # color = 'grey',
        #         label = list(set(sc_types))[idx], s = 50, alpha = 0.6)
    # ax1.legend(loc='upper left', prop={'size': 4})
    
    tmp_handles, tmp_labels = ax1.get_legend_handles_labels()
    order_index = sorted(range(len(tmp_labels)), key=tmp_labels.__getitem__)
    handles = [tmp_handles[idx] for idx in order_index]
    labels = [tmp_labels[idx] for idx in order_index]
    ax1.set_xlim(left = min(sc_x) - horizontal_margin, 
            right = max(sc_x) + horizontal_margin)
    ax1.set_ylim(bottom = min(sc_y) - vertical_margin, 
            top = max(sc_y) + vertical_margin)
    ax1.set_title('Original data \n')
    ax1.set_ylabel('tSNE element 2')
    ax1.set_xlabel('tSNE element 1')
    ax1.set_xticklabels([])


    # plot the online NMF
    ax3 = fig.add_subplot(132)
    ax3.scatter(sc_x, sc_y, color = 'grey')
    # for idx in range(k_cluster):
    for centroid_idx, color_idx in enumerate(onmf_centroid_order_list):
        ax3.scatter(non_cvx_centroid_x[centroid_idx], non_cvx_centroid_y[centroid_idx], 
                color = centroid_color_list[color_idx], 
                marker = marker_list[color_idx],
                alpha = 0.9, label = non_cvx_centroid_types[centroid_idx])
    # ax3.set_title('{}\nAccuracy: {:.4f}, AMI: {:.4f}'.format('online_NMF', acc_noncvx, AMI_noncvx))
    ax3.set_xlim(left = min(sc_x) - horizontal_margin, 
            right = max(sc_x) + horizontal_margin)
    ax3.set_ylim(bottom = min(sc_y) - vertical_margin, 
            top = max(sc_y) + vertical_margin)
    ax3.set_title('online MF \n accuracy:{:.4f}'.format(acc_omf))
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xlabel('tSNE element 1')
    # ax3.set_xlim(left = (min(sc_x) - (max(sc_x) - min(sc_x))/1))
    # ax3.legend(loc = 'upper left', prop={'size': 4})
    

    # plot cvx online NMF Rr
    ax4 = fig.add_subplot(133)
    ax4.scatter(sc_x, sc_y, color = 'grey')
    # for idx in range(k_cluster):
    for centroid_idx, color_idx in enumerate(ocnmf_centroid_order_list):
        x_hat_start_idx = int(X_hat_index[centroid_idx])
        x_hat_end_idx = int(X_hat_index[centroid_idx + 1])
        ax4.scatter(X_hat_Rr_x[x_hat_start_idx:x_hat_end_idx], 
                X_hat_Rr_y[x_hat_start_idx:x_hat_end_idx], 
                facecolor = color_list[color_idx], 
                # label = list(set(X_hat_types))[idx], 
                marker = 'x', s = 50, alpha = 0.3)
    

    for centroid_idx, color_idx in enumerate(ocnmf_centroid_order_list):
        ax4.scatter(centroid_x[centroid_idx], centroid_y[centroid_idx], 
                color = centroid_color_list[color_idx], 
                marker = marker_list[color_idx],
                alpha = 0.9, label = centroid_types[centroid_idx])
    
    ax4.set_xlim(left = min(sc_x) - horizontal_margin, 
            right = max(sc_x) + horizontal_margin)
    ax4.set_ylim(bottom = min(sc_y) - vertical_margin, 
            top = max(sc_y) + vertical_margin)
    ax4.set_title('online cvxMF ($R_r$) \n accuracy:{:.4f}'.format(acc_Rr))
    ax4.set_xlabel('tSNE element 1')
    # ax4.set_ylabel('tSNE element 2')
    # ax4.set_xlim(left = (min(sc_x) - (max(sc_x) - min(sc_x))/1))
    # ax4.legend(loc = 'upper left', prop={'size': 4})

    # plot cvx online NMF Ru
    # ax5 = fig.add_subplot(224)
    # ax5.scatter(sc_x, sc_y, color = 'grey')
    # # for idx in range(k_cluster):
    # for centroid_idx, color_idx in enumerate(ocnmf_Ru_centroid_order_list):
    #     x_hat_start_idx = int(X_hat_index_Ru[centroid_idx])
    #     x_hat_end_idx = int(X_hat_index_Ru[centroid_idx + 1])
    #     ax5.scatter(X_hat_Ru_x[x_hat_start_idx:x_hat_end_idx], X_hat_Ru_y[x_hat_start_idx:x_hat_end_idx], 
    #             facecolor = color_list[color_idx], 
    #             # label = list(set(X_hat_types))[idx], 
    #             marker = 'x', s = 50, alpha = 0.3)
    # 

    # for centroid_idx, color_idx in enumerate(ocnmf_Ru_centroid_order_list):
    #     ax5.scatter(centroid_Ru_x[centroid_idx], centroid_Ru_y[centroid_idx], 
    #             color = centroid_color_list[color_idx], 
    #             marker = marker_list[color_idx],
    #             alpha = 0.9, label = centroid_types[centroid_idx])
    # 
    # ax5.set_xlim(left = min(sc_x) - horizontal_margin, 
    #         right = max(sc_x) + horizontal_margin)
    # ax5.set_ylim(bottom = min(sc_y) - vertical_margin, 
    #         top = max(sc_y) + vertical_margin)
    # ax5.set_title('online cvxMF $R_u$ with \n bases and representative sets')
    # ax5.set_xlabel('tSNE element 1')
    # ax5.set_yticklabels([])
    # # ax4.set_xlim(left = (min(sc_x) - (max(sc_x) - min(sc_x))/1))
    # # ax4.legend(loc = 'upper left', prop={'size': 4})


    ax1.grid(color = 'grey', alpha = 0.4)
    ax3.grid(color = 'grey', alpha = 0.4)
    ax4.grid(color = 'grey', alpha = 0.4)
    # ax5.grid(color = 'grey', alpha = 0.4)
    # plt.tight_layout()

    return fig

def plot_eigen_images(nrows, ncols, centroid_mat, 
        # col_pixels = 192, row_pixels = 168 ):
        col_pixels = 28, row_pixels = 28 ):
    '''
    plot all eigen images in matrix nrows *  ncols
    '''
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols,
            sharex = True, sharey = True,
            dpi = 150)
    num_cluster = centroid_mat.shape[0]
    for idx in range(nrows * ncols):
        if idx < num_cluster:
            fig_np = centroid_mat[idx].reshape(col_pixels, row_pixels)
            row, col = np.unravel_index(idx, (nrows, ncols))
            ax[row][col].imshow(fig_np)
            ax[row][col].set_axis_off()

    plt.tight_layout()
    plt.subplots_adjust(wspace = -0.7, hspace = 0.1)
    return fig
