#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import lmdb
import ctypes
import shutil
import datetime
import numpy as np
import multiprocessing

#https://github.com/BVLC/caffe/issues/861#issuecomment-70124809
import matplotlib 
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

sys.path.append(os.path.join(g_caffe_install_path, 'python'))
import caffe

try:
    # Python3 will most likely not be able to load protobuf
    from caffe.proto import caffe_pb2
except:
    if sys.version_info >= (3, 0):
        print("Failed to include caffe_pb2, things might go wrong!")
    else:
        raise

print datetime.datetime.now().time(), '- Allocating memory...'
syn_images_num = len([line.strip() for line in open(g_syn_images_train_val_split, 'r')])
env = lmdb.open(g_pool5_lmdb, readonly=True)
datum_array = np.empty(0)
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value)
        datum_array = caffe.io.datum_to_array(datum)
        break;
print datetime.datetime.now().time(), '- the shape of the datum is:', datum_array.shape

shared_array_base = multiprocessing.Array(ctypes.c_double, syn_images_num*datum_array.size)
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape([syn_images_num]+list(datum_array.shape))

def image_pair_to_string(pair_info, def_param=shared_array):
    array_1_id = pair_info[0]
    array_2_id = pair_info[1]
    similar_or_not = pair_info[2]
    array_1 = shared_array[array_1_id, :]
    array_2 = shared_array[array_2_id, :]
    
    array = np.concatenate((array_1, array_2), axis=0)
    datum = caffe.io.array_to_datum(array)
    datum.label = similar_or_not
    return datum.SerializeToString() 

def string_to_array(datum_string_idx, def_param=shared_array):
    datum_string = datum_string_idx[0]
    idx = datum_string_idx[1]
    datum = caffe_pb2.Datum()
    datum.ParseFromString(datum_string)
    shared_array[idx, :] = caffe.io.datum_to_array(datum)

pool = multiprocessing.Pool(g_gen_siamese_lmdb_thread_num)
report_step = 1000;

print  datetime.datetime.now().time(), '- Loading feature from %s...'%(g_pool5_lmdb)
loaded_count = 0
cache = []
convert_count = 512
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        cache.append((value, (int)(key)))
        if (len(cache) == convert_count or loaded_count == syn_images_num-1):
            pool.map(string_to_array, cache)
            del cache[:]
        if(loaded_count%report_step == 0):
            print datetime.datetime.now().time(), '-', loaded_count, 'of', syn_images_num, 'features loaded!'
        loaded_count = loaded_count + 1
env.close()

print datetime.datetime.now().time(), '- Starting generating pairs lmdbs...'

if os.path.exists(g_pairs_pool5_lmdb_train):
    shutil.rmtree(g_pairs_pool5_lmdb_train)
env_train = lmdb.open(g_pairs_pool5_lmdb_train, map_size=int(1e12))
if os.path.exists(g_pairs_pool5_lmdb_val):
    shutil.rmtree(g_pairs_pool5_lmdb_val)
env_val = lmdb.open(g_pairs_pool5_lmdb_val, map_size=int(1e12))

train_val_split = [int(line.strip()) for line in open(g_syn_images_pairs_train_val_split, 'r')]
image_pair_list = [[(int)(value) for value in line.strip().split(' ')] for line in open(g_syn_images_pairs_filelist, 'r')]

cache_train_id = []
cache_train_image_pair = []
cache_val_id = []
cache_val_image_pair = []

txn_commit_count = 512
for idx, image_pair in enumerate(image_pair_list):
    key_idx = '{:0>10d}'.format(idx)
    if train_val_split[idx]:
        cache_train_id.append(key_idx)
        cache_train_image_pair.append(image_pair)
        if (len(cache_train_id) == txn_commit_count or idx == len(train_val_split)-1):
            train_string_list = pool.map(image_pair_to_string, cache_train_image_pair)
            with env_train.begin(write=True) as txn_train:
                for i in range(len(cache_train_id)):
                    txn_train.put(cache_train_id[i], train_string_list[i])
            del cache_train_id[:]
            del cache_train_image_pair[:]
    else:
        cache_val_id.append(key_idx)
        cache_val_image_pair.append(image_pair)
        if (len(cache_val_id) == txn_commit_count or idx == len(train_val_split)-1):
            val_string_list = pool.map(image_pair_to_string, cache_val_image_pair)
            with env_val.begin(write=True) as txn_val:
                for i in range(len(cache_val_id)):
                    txn_val.put(cache_val_id[i], val_string_list[i])
            del cache_val_id[:]
            del cache_val_image_pair[:]
            
    if(idx%report_step == 0):
        print datetime.datetime.now().time(), '-', idx, 'of', len(train_val_split), 'processed!' 

env_train.close()
env_val.close()
