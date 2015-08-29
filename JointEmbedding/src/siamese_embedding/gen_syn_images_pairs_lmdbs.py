#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import lmdb
import shutil
import datetime
from multiprocessing import Pool

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
        
def image_pair_to_string(pair_info):
    image_pair = pair_info[0]
    id_to_data = pair_info[1]
    pair_idx = pair_info[2]
    datum_1_id = image_pair[0]
    datum_2_id = image_pair[1]
    datum_1 = id_to_data[datum_1_id]
    datum_2 = id_to_data[datum_2_id]
    
    datum = caffe_pb2.Datum()
    datum.channels = 2
    datum.height = datum1.height
    datum.width = 1
    datum.label = pair_idx
    assert(len(datum_1.float_data) == len(datum_2.float_data))
    for data in datum_1.float_data:
        datum.float_data.append(data)
    for data in datum_2.float_data:
        datum.float_data.append(data)    
    return datum.SerializeToString()  
    
def string_to_datum(datum_string):
    datum = caffe_pb2.Datum()
    datum.ParseFromString(datum_string)
    return datum 
        
pool = Pool(g_gen_siamese_lmdb_thread_num)
        


report_step = 10000;

print 'Loading feature from %s...'%(g_pool5_lmdb)
id_to_data = dict()

env = lmdb.open(g_pool5_lmdb, readonly=True)
idx = 0
cache_id = []
cache_datum_string = []
convert_count = 512
with env.begin() as txn:
    cursor = txn.cursor()
    print len(cursor)
    for key, value in cursor:
        cache_id.append((int)(key))
        cache_datum_string.append(value)
        if (len(cache_datum_string) == convert_count):
            datum_list = pool.map(string_to_datum, cache_datum_string)
            id_to_data.update(dict(zip(cache_id, datum_list)))
            cache_id.clear()
            cache_datum_string.clear()
            
        if(idx%report_step == 0):
            print datetime.datetime.now().time(), '-', idx, 'of', len(train_val_split), ' features loaded!'
        idx = idx + 1    

if (len(cache_datum_string) != 0):
    datum_list = pool.map(string_to_datum, cache_datum_string)
    id_to_data.update(dict(zip(cache_id, datum_list)))
    cache_id.clear()
    cache_datum_string.clear()
env.close()

if os.path.exists(g_pairs_pool5_lmdb_train):
    shutil.rmtree(g_pairs_pool5_lmdb_train)
env_train = lmdb.open(g_pairs_pool5_lmdb_train, map_size=int(1e12))
if os.path.exists(g_pairs_pool5_lmdb_val):
    shutil.rmtree(g_pairs_pool5_lmdb_val)
env_val = lmdb.open(g_pairs_pool5_lmdb_val, map_size=int(1e12))

train_val_split = [int(line.strip()) for line in open(g_syn_images_pairs_train_val_split, 'r')]
image_pair_list = [[(int)(value) for value in line.strip().split(' ')] for line in open(g_syn_images_pairs_filelist, 'r')]

cache_train_id = []
cache_traing_image_pair = []
cache_val_id = []
cache_val_image_pair = []

txn_commit_count = 512
for idx, image_pair in image_pair_list:
    key_idx = '{:0>10d}'.format(idx)
    if train_val_split[idx]:
        cache_train_id.append(key_idx)
        cache_traing_image_pair.append((image_pair, id_to_data, idx))
        if (len(cache_train) == txn_commit_count or idx == len(train_val_split)-1):
            train_string_list = pool.map(image_pair_to_string, cache_traing_image_pair)
            with env_train.begin(write=True) as txn_train:
                for i in range(cache_train_id):
                    txn_train.put(cache_train_id[i], train_string_list[i])
            cache_train_id.clear()
            cache_traing_image_pair.clear()
    else:
        cache_val_id.append(key_idx)
        cache_valg_image_pair.append((image_pair, id_to_data))
        if (len(cache_val) == txn_commit_count or idx == len(val_val_split)-1):
            val_string_list = pool.map(image_pair_to_string, cache_valg_image_pair)
            with env_val.begin(write=True) as txn_val:
                for i in range(cache_val_id):
                    txn_val.put(cache_val_id[i], val_string_list[i])
            cache_val_id.clear()
            cache_valg_image_pair.clear()
            
    if(idx%report_step == 0):
        print datetime.datetime.now().time(), '-', idx, 'of', len(train_val_split), 'processed!' 

env_train.close()
env_val.close()
