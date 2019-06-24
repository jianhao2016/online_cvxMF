#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np
import time

def np_mat(a, b):
    print(a.shape, b.shape)
    t1 = time.time()
    c = a @ b
    t2 = time.time()
    print('np cost {:.4f}s'.format(t2 - t1))


if __name__ == '__main__':
    a = np.load('/data/jianhao/tmp_test/a_mat_full.npy')
    b = np.load('/data/jianhao/tmp_test/b_mat_full.npy')

    np_mat(a, b)

