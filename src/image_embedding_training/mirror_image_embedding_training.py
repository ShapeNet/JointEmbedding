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

shapenet_synset_set_handle_mirror = g_shapenet_synset_set_handle + '_' + g_mirror_name;
filelist_mirror = os.path.join(g_data_folder, 'image_embedding/syn_images_filelist'+shapenet_synset_set_handle_mirror+'.txt')
filelist_mirror_train = os.path.join(g_data_folder, 'image_embedding/syn_images_filelist'+shapenet_synset_set_handle_mirror+'_train.txt')
filelist_mirror_val = os.path.join(g_data_folder, 'image_embedding/syn_images_filelist'+shapenet_synset_set_handle_mirror+'_val.txt')
imageid2shapeid_mirror = os.path.join(g_data_folder, 'image_embedding/syn_images_imageid2shapeid'+shapenet_synset_set_handle_mirror+'.txt')
imageid2shapeid_mirror_train = os.path.join(g_data_folder, 'image_embedding/syn_images_imageid2shapeid'+shapenet_synset_set_handle_mirror+'_train.txt')
imageid2shapeid_mirror_val = os.path.join(g_data_folder, 'image_embedding/syn_images_imageid2shapeid'+shapenet_synset_set_handle_mirror+'_val.txt')
train_val_split_mirror = os.path.join(g_data_folder, 'image_embedding/syn_images_train_val_split'+shapenet_synset_set_handle_mirror+'.txt')

shape_list_mirror_mapping_filename = os.path.join(g_data_folder, 'shape_list_mapping'+shapenet_synset_set_handle_mirror+'.txt')
print 'Loading shape list mapping from %s...'%(shape_list_mirror_mapping_filename)
shape_list_mirror_mapping = [bool(int(line.strip())) for line in open(shape_list_mirror_mapping_filename, 'r')]
shapeid_mapping = []
shape_count = 0
for item in shape_list_mirror_mapping:
    if item:
        shapeid_mapping.append(shape_count)
        shape_count = shape_count + 1
    else:
        shapeid_mapping.append(-1)

print 'Loading original imageid2shapeid from %s...'%(g_syn_images_imageid2shapeid)
imageid2shapeid_original = [int(line.strip()) for line in open(g_syn_images_imageid2shapeid, 'r')]

print 'Loading original filelist from %s...'%(g_syn_images_filelist)
filelist_original = [line.strip() for line in open(g_syn_images_filelist, 'r')]

print 'Loading original train_val_split from %s...'%(g_syn_images_train_val_split)
train_val_split_original = [int(line.strip()) for line in open(g_syn_images_train_val_split, 'r')]

print 'Writing mirror files...'
filelist_mirror_file = open(filelist_mirror, 'w')
filelist_mirror_train_file = open(filelist_mirror_train, 'w')
filelist_mirror_val_file = open(filelist_mirror_val, 'w')
imageid2shapeid_mirror_file = open(imageid2shapeid_mirror, 'w')
imageid2shapeid_mirror_train_file = open(imageid2shapeid_mirror_train, 'w')
imageid2shapeid_mirror_val_file = open(imageid2shapeid_mirror_val, 'w')
train_val_split_mirror_file = open(train_val_split_mirror, 'w')

imageid2shapeid_mirror = []
for idx, shapeid_original in enumerate(imageid2shapeid_original):
    if shape_list_mirror_mapping[shapeid_original]:
        filelist_mirror_file.write(filelist_original[idx]+'\n')
        shapeid_mirror = shapeid_mapping[shapeid_original]
        assert(shapeid_mirror != -1)
        imageid2shapeid_mirror_file.write(str(shapeid_mirror)+'\n')
        imageid2shapeid_mirror.append(shapeid_mirror)
        train_val = train_val_split_original[idx]
        train_val_split_mirror_file.write(str(int(train_val))+'\n')
        if train_val:
            filelist_mirror_train_file.write(filelist_original[idx]+'\n')
            imageid2shapeid_mirror_train_file.write(str(shapeid_mirror)+'\n')
        else:
            filelist_mirror_val_file.write(filelist_original[idx]+'\n')
            imageid2shapeid_mirror_val_file.write(str(shapeid_mirror)+'\n')          

filelist_mirror_file.close()
filelist_mirror_train_file.close()
filelist_mirror_val_file.close()
imageid2shapeid_mirror_file.close()
imageid2shapeid_mirror_train_file.close()
imageid2shapeid_mirror_val_file.close()
train_val_split_mirror_file.close()

imageid_mapping = []
image_count = 0
for idx, shapeid_original in enumerate(imageid2shapeid_original):
    if shape_list_mirror_mapping[shapeid_original]:
        imageid_mapping.append(image_count)
        shape_count = shape_count + 1
    else:
        imageid_mapping.append(-1)
        
def map_global_idx(datum_string, def_param=imageid_mapping):
    datum = caffe_pb2.Datum()
    datum.ParseFromString(datum_string)
    datum.label = imageid_mapping[datum.label]
    assert(datum.label != -1)
    return datum.SerializeToString()

pool = Pool(g_extract_feat_thread_num)

env_original = lmdb.open(g_pool5_lmdb, readonly=True)
pool5_lmdb_mirror = os.path.join(g_data_folder, 'image_embedding/syn_images_pool5_lmdb'+shapenet_synset_set_handle_mirror+'_'+g_network_architecture_name)
if os.path.exists(pool5_lmdb_mirror):
    shutil.rmtree(pool5_lmdb_mirror)
env_mirror = lmdb.open(pool5_lmdb_mirror, map_size=int(1e12))
print 'Writing lmdb to %s...'%(pool5_lmdb_mirror)

txn_commit_count = 512
report_step = 10000;
idx_original = 0
idx_mirror = 0
cache_key_mirror = []
cache_value_original = []
with env_original.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        if shape_list_mirror_mapping[imageid2shapeid_original[idx_original]]:
            key_mirror = '{:0>10d}'.format(idx_mirror)
            cache_key_mirror.append(key_mirror)
            cache_value_original.append(value)
            if(idx_mirror%report_step == 0):
                print datetime.datetime.now().time(), '-', idx_mirror, 'of', len(imageid2shapeid_mirror), 'processed!'
            idx_mirror = idx_mirror + 1
            
        if (len(cache_key_mirror) == txn_commit_count or idx_original == len(imageid2shapeid_original)-1):
            cache_value_mirror = pool.map(map_global_idx, cache_value_original)
            with env_mirror.begin(write=True) as txn_mirror:
                for idx in range(len(cache_key_mirror)):
                    txn_mirror.put(cache_key_mirror[idx], cache_value_mirror[idx])
            del cache_key_mirror[:]
            del cache_value_original[:]
            
        idx_original = idx_original + 1
