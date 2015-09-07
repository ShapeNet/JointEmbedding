#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
from multiprocessing import Pool

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *
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

print 'Loading imageid2shapeid file from %s...'%(args.imageid2shapeid)
imageid2shapeid_mapping = [int(line.strip()) for line in open(args.imageid2shapeid, 'r')]
        
def convert_imageid2shapeid(datum_string, def_param=imageid2shapeid_mapping):
    assert(imageid_output != -1)
    datum = caffe_pb2.Datum()
    datum.ParseFromString(datum_string)
    datum.label = imageid2shapeid_mapping[datum.label]
    return datum.SerializeToString()

pool = Pool(g_extract_feat_thread_num)

env_input = lmdb.open(args.input_lmdb, readonly=True)
if os.path.exists(args.output_lmdb):
    shutil.rmtree(args.output_lmdb)
env_output = lmdb.open(args.output_lmdb, map_size=int(1e12))

txn_commit_count = 512
report_step = 10000;
idx = 0
cache_key_output = []
cache_value_input = []
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        key_output = '{:0>10d}'.format(idx)
        assert(key == key_output)
        cache_key_output.append(key_output)
        cache_value_input.append(value)
        if (len(cache_key_output) == txn_commit_count or idx == len(imageid2shapeid_mapping)-1):
            cache_value_output = pool.map(convert_imageid2shapeid, cache_value_input)
            with env_output.begin(write=True) as txn_output:
                for idx in range(len(cache_key_output)):
                    txn_output.put(cache_key_output[idx], cache_value_output[idx])
            del cache_key_output[:]
            del cahce_value_input[:]
            
        if(idx%report_step == 0):
            print datetime.datetime.now().time(), '-', idx, 'of', len(imageid2shapeid_mapping), 'processed!'
        idx = idx + 1
