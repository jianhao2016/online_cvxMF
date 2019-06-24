#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script is to load the 10x Genomics data
"""

import csv
import os
import scipy.io
import numpy as np
import pandas as pd
import ipdb
import pickle
from scipy.sparse import coo_matrix, hstack

genome = 'hg19'
# cell_lists = ['cd19_b_cells', 'cd56_natural_killer_cells', 'regulatory_t_cells']
# root_dir = '/data/jianhao/scRNA_seq'
data_dir = '/data/shared/jianhao/10xGenomics_scRNA/pandasDF'
root_dir = '/data/shared/jianhao/10xGenomics_scRNA/bash_downloaded/'
cell_lists = [ctype for ctype in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, ctype))]
# ipdb.set_trace()
human_matirces_dir = [os.path.join(root_dir, p, genome) for p in cell_lists]

num_cells_each_cluster = -1
shrinked_mat_list = []
cell_type_list = []
gene_list = []
np.random.seed(42)

for data_path, cell_type in zip(human_matirces_dir, cell_lists):
    if os.path.isdir(data_path):
        mat_tmp = scipy.io.mmread(os.path.join(data_path, 'matrix.mtx'))
        mat_tmp = mat_tmp.tocsc()

        num_genes, num_cells = mat_tmp.shape
        total_elements = num_cells * num_genes
        print('cell type: ', cell_type)
        print('non zero elements: {:.6f}'.format(len(mat_tmp.nonzero()[0])/total_elements))
        print('mean counts: {:.6f}'.format(mat_tmp.mean()))
        print('num of genes:', num_genes, ', num of cells:', num_cells)

        if num_cells_each_cluster != -1:
            selected_cells = np.random.permutation(num_cells)[:num_cells_each_cluster]
            shrinked_mat = mat_tmp[:, selected_cells]
        else:
            # when num_cells_each_cluster = -1, 
            # select all the cells.
            selected_cells = np.arange(num_cells)
            shrinked_mat = mat_tmp
        print('sum of first 5 cells:', shrinked_mat[:, :5].sum(axis = 0))
        shrinked_mat_list.append(shrinked_mat)

        gene_path_tmp = os.path.join(data_path, 'genes.tsv')
        gene_ids_tmp = [row[0] for row in csv.reader(open(gene_path_tmp), delimiter='\t')]
        print('number of genes being measure:', len(gene_ids_tmp))
        gene_list = gene_ids_tmp

        barcodes_path = os.path.join(data_path, 'barcodes.tsv')
        barcodes_tmp = [row[0] for row in csv.reader(open(barcodes_path), delimiter='\t')]
        shrinked_barcodes_tmp = np.array(barcodes_tmp)[selected_cells]
        print('number of cells in cluster: ', len(shrinked_barcodes_tmp))
        cell_type_list += [cell_type] * len(shrinked_barcodes_tmp)
        print('-'*7)

    else:
        print('{} is not a valid path!'.format(data_path))
        #print(sum(mat_tmp.toarray()))

# with open('/home/jianhao2/expr_inference/landmark_gene_name_list.pickle', 'rb') as f:
#     lm_list = pickle.load(f)
# 
# i = 0
# shared_index = []
# for g in lm_list:
#     if g in gene_list:
#         i += 1
#         shared_index.append(gene_list.index(g))

# shared_genes = np.array(gene_list)[shared_index]
# tmp_list = [m.tocsr()[shared_index, :] for m in shrinked_mat_list]
# shrinked_mat_list = tmp_list
shared_genes = gene_list

dense_mat = hstack(shrinked_mat_list).todense()
print('size of stack dense mat: ', dense_mat.shape)
# np.save(os.path.join(root_dir, 'dense_data_500'), dense_mat)
X = dense_mat.T
df = pd.DataFrame(data = X, columns = list(shared_genes))
k_cluster = len(cell_lists)
with open(os.path.join(data_dir, 'df_feature_column_{}_clusters_{}'.format(k_cluster,
                        num_cells_each_cluster)), 'wb') as f:
    pickle.dump(list(shared_genes), f)

df['label'] = cell_type_list
print('Size of data frame', df.shape)
print(df['label'].shape)
# print(df['label'][:10], df['label'][500:510], df['label'][1000:1010])
df.to_pickle(os.path.join(data_dir, 'pandas_dataframe_{}_clusters_{}'.format(k_cluster, 
                        num_cells_each_cluster)))
