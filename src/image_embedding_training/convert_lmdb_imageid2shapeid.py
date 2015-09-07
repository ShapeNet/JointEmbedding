#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import lmdb
import shutil
import datetime
import argparse
from multiprocessing import Pool

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

parser = argparse.ArgumentParser(description="Convert imageid in LMDB into shapeid.")
parser.add_argument('--input_lmdb', help='Path to input LMDB.', required=True)
parser.add_argument('--output_lmdb', help='Path to output LMDB.', required=True)
parser.add_argument('--imageid2shapeid', help='Path to imageid2shapeid file.', required=True)
args = parser.parse_args()

print 'Loading imageid2shapeid file from %s...'%(args.imageid2shapeid)
imageid2shapeid_mapping = [int(line.strip()) for line in open(args.imageid2shapeid, 'r')]
        
def convert_imageid2shapeid(datum_string, def_param=imageid2shapeid_mapping):
    datum = caffe_pb2.Datum()
    datum.ParseFromString(datum_string)
    datum.label = imageid2shapeid_mapping[datum.label]
    assert(datum.label != -1)
    return datum.SerializeToString()

pool = Pool(g_extract_feat_thread_num)

env_input = lmdb.open(args.input_lmdb, readonly=True)
if os.path.exists(args.output_lmdb):
    shutil.rmtree(args.output_lmdb)
env_output = lmdb.open(args.output_lmdb, map_size=int(1e12))

txn_commit_count = 512
report_step = 10000;
image_count = 0
cache_key_output = []
cache_value_input = []
with env_input.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        key_output = '{:0>10d}'.format(image_count)
        cache_key_output.append(key_output)
        cache_value_input.append(value)
        if (len(cache_key_output) == txn_commit_count or image_count == len(imageid2shapeid_mapping)-1):
            cache_value_output = pool.map(convert_imageid2shapeid, cache_value_input)
            with env_output.begin(write=True) as txn_output:
                for idx in range(len(cache_key_output)):
                    txn_output.put(cache_key_output[idx], cache_value_output[idx])
            del cache_key_output[:]
            del cache_value_input[:]
            
        if(image_count%report_step == 0):
            print datetime.datetime.now().time(), '-', image_count, 'of', len(imageid2shapeid_mapping), 'processed!'
        image_count = image_count + 1
