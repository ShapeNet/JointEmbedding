#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import lmdb
import datetime
from multiprocessing import Pool

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *
sys.path.append(os.path.join(g_caffe_install_path, 'python'))
import caffe

report_step = 10000;

lmdb_names = [g_pool5_lmdb_val, g_pool5_lmdb_train, g_shape_embedding_lmdb_val, g_shape_embedding_lmdb_train]
keys_dict = dict()
labels_dict = dict()

def extract_label(datum_string):
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(datum_string)
    return datum.label 
    
convert_count = 512
#pool = Pool(g_extract_feat_thread_num)

for lmdb_name in lmdb_names:
    env = lmdb.open(lmdb_name, map_size=int(1e12))
    keys = []
    labels = []
    idx = 0
    print datetime.datetime.now().time(), '-', 'loading lmdb from', lmdb_name
    with env.begin() as txn:
        cursor = txn.cursor()
        string_cache = []
        for key, value in cursor:
            keys.append(key)
            #string_cache.append(value)
            if (len(string_cache) == convert_count):
                labels_pool = pool.map(extract_label, string_cache)
                labels.extend(labels_pool)
                del string_cache[:]
                
            if(idx%report_step == 0):
                print datetime.datetime.now().time(), '-', idx, 'processed!'
            idx = idx + 1  
    env.close()
    
    if (len(string_cache) != 0):
        labels_pool = pool.map(extract_label, string_cache)
        labels.extend(labels_pool)
    keys_dict[lmdb_name] = keys
    labels_dict[lmdb_name] = labels

    with open(lmdb_name+'.txt', 'w') as keys_file:
        for key in keys:
            keys_file.write(key+'\n')

assert(keys_dict[g_pool5_lmdb_val] == keys_dict[g_shape_embedding_lmdb_val])
assert(keys_dict[g_pool5_lmdb_train] == keys_dict[g_shape_embedding_lmdb_train])
