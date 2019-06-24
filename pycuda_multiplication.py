#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script contains some cuda matrix calculation.
"""

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
import skcuda.misc as misc
import time
from functools import reduce

def fast_matmul(x, y, x_type, y_type):
    '''
    use pycuda to compute c = a * b
    '''
    linalg.init()
    a_gpu = gpuarray.to_gpu(x.astype(x_type))
    a_t_gpu = gpuarray.to_gpu(x.T.copy().astype(x_type))
    b_gpu = gpuarray.to_gpu(y.astype(y_type))
    # row_sum = gpuarray.zeros(shape = x[0].shape, dtype = x_type)
    row_sum = 0
    # a = np.asarray(x, x_type)
    # b = np.asarray(y, y_type)
    # a_gpu = gpuarray.to_gpu(a)
    # b_gpu = gpuarray.to_gpu(b)

    t1_inside = time.time()
    c_gpu = linalg.dot(a_gpu, b_gpu)
    for a_i in a_gpu:
        # row_sum = misc.add(row_sum, a_i)
        row_sum += a_i
        gg = linalg.dot(a_gpu, b_gpu)
        gg = linalg.dot(a_i, a_i)
        gg = reduce(linalg.dot, (a_gpu, b_gpu, b_gpu, b_gpu))
        # tmp1, tmp2 = linalg.dot(a_gpu, b_gpu), linalg.dot(b_gpu, b_gpu)
        z_gpu = a_gpu.copy()
    tmp = a_t_gpu
    # print('x.T\n', x.T)
    # print('tmp\n', tmp)
    # print('x = a_gpu: ', np.allclose(x, a_gpu.get()))
    # print('x.T = tmp: ', np.allclose(x.T, tmp.get()))

    a_prod = linalg.dot(a_gpu, tmp)
    t2_inside = time.time()
    print('inside cost {:.4f}s'.format(t2_inside - t1_inside))

    a = np.random.randint(-5, 5, (3, 4)).astype(np.float32)
    a_gpu = gpuarray.to_gpu(a)
    norm_gpu = linalg.norm(a_gpu)
    print('is norm right?', np.linalg.norm(a) == norm_gpu)
    a_gpu = abs(a_gpu)
    column_sum = misc.sum(a_gpu, axis = 0)
    column_sum = column_sum.reshape((1, -1))
    all_one_gpu = gpuarray.to_gpu(np.ones((3, 1), np.float32))
    div_mat_gpu = linalg.dot(all_one_gpu, column_sum)


    norm_1 = a_gpu / (div_mat_gpu + 1e-3)
    
    print(a_gpu)
    print(column_sum)
    print(column_sum.shape)
    print(norm_1)
    # abs_a = a_gpu.__abs__()
    # print(a)
    # print(abs_a)
    # c = abs_a + a_gpu
    # print(repr(c))
    # print(type(c))
    # c = 1/2 * c
    # print(a_gpu, c)
    return c_gpu.get(), a_prod.get(), row_sum.get()


def np_matmul(x, y):
    row_sum = np.zeros_like(x[0])
    t1 = time.time()
    z = x @ y
    for x_i in x:
        row_sum += x_i
        gg = x @ y
        gg = reduce(np.dot, (x, y, y, y))
        z_gpu = x.copy()
    x_prod = x @ x.T

    t2 = time.time()
    print('a @ b cost {:.4f}s'.format(t2 - t1))
    return z, x_prod, row_sum

def fast_add(x, y, x_type, y_type):
    '''
    use pycuda to compute c = a * b
    '''
    linalg.init()
    a_gpu = gpuarray.to_gpu(x.astype(x_type))
    b_gpu = gpuarray.to_gpu(y.astype(y_type))

    t1_inside = time.time()
    # c_gpu = misc.add(a_gpu, b_gpu)
    c_gpu = a_gpu + b_gpu
    t2_inside = time.time()
    print('inside cost {:.4f}s'.format(t2_inside - t1_inside))
    return c_gpu.get()

def np_add(x, y):
    t1 = time.time()
    z = x + y
    t2 = time.time()
    print('a + b cost {:.4f}s'.format(t2 - t1))

if __name__ == '__main__':
    # x = np.load('/data/jianhao/tmp_test/a_mat_full.npy')
    # y = np.load('/data/jianhao/tmp_test/b_mat_full.npy')
    x = np.random.randn(50, 30).astype(np.float32)
    y = np.random.randn(30, 30).astype(np.float32)
    
    print(x.shape, y.shape)

    t1 = time.time()
    zz_cuda, x_prod_cuda, row_sum_cuda = fast_matmul(x, y, np.float64, np.float64)
    t2 = time.time()
    print('pycuda cost {:.4f}s'.format(t2 - t1))

    zz, x_prod, row_sum = np_matmul(x, y)
    print('-' * 7)

    print('x * y:', np.allclose(zz, zz_cuda))
    print('x * x.T:', np.allclose(x_prod, x_prod_cuda))
    print('row sum:', np.allclose(row_sum, row_sum_cuda))

    
    # t1 = time.time()
    # zz = fast_add(y, y, np.float32, np.float32)
    # t2 = time.time()
    # print('pycuda cost {:.4f}s'.format(t2 - t1))

    # np_add(y, y)


