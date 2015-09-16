#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import lmdb
import shutil
import datetime
import numpy as np

#https://github.com/BVLC/caffe/issues/861#issuecomment-70124809
import matplotlib
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

sys.path.append(os.path.join(g_caffe_install_path, 'python'))
import caffe

train_val_split = [int(line.strip()) for line in open(g_syn_images_train_val_split, 'r')]
imageid2shapeid = [int(line.strip()) for line in open(g_syn_images_imageid2shapeid, 'r')]
embedding_space = [[float(value) for value in line.strip().split(' ')] for line in open(g_shape_embedding_space_file_txt, 'r')]
assert(len(embedding_space[0]) == g_shape_embedding_space_dimension)
embedding_space_arrays = [np.array(embedding).reshape(g_shape_embedding_space_dimension, 1, 1) for embedding in embedding_space]
embedding_space_strings = [caffe.io.array_to_datum(array, idx).SerializeToString() for idx, array in enumerate(embedding_space_arrays)]

if os.path.exists(g_shape_embedding_lmdb_train):
    shutil.rmtree(g_shape_embedding_lmdb_train)
env_train = lmdb.open(g_shape_embedding_lmdb_train, map_size=int(1e12))
if os.path.exists(g_shape_embedding_lmdb_val):
    shutil.rmtree(g_shape_embedding_lmdb_val)
env_val = lmdb.open(g_shape_embedding_lmdb_val, map_size=int(1e12))

cache_train = dict()
cache_val = dict()
txn_commit_count = 512

report_step = 10000;
for idx, train_val in enumerate(train_val_split):
    key = '{:0>10d}'.format(idx)
    value = embedding_space_strings[imageid2shapeid[idx]]
    if train_val == 1:
        cache_train[key] = value
    elif train_val == 0:
        cache_val[key] = value
        
    if (len(cache_train) == txn_commit_count or idx == len(train_val_split)-1):
        with env_train.begin(write=True) as txn_train:
            for k, v in sorted(cache_train.iteritems()):
                txn_train.put(k, v)
        cache_train.clear()
    if (len(cache_val) == txn_commit_count or idx == len(train_val_split)-1):
        with env_val.begin(write=True) as txn_val:
            for k, v in sorted(cache_val.iteritems()):
                txn_val.put(k, v)
        cache_val.clear()
        
    if(idx%report_step == 0):
        print datetime.datetime.now().time(), '-', idx, 'of', len(train_val_split), 'processed!'
        
env_train.close()
env_val.close()
