#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import math
import lmdb
import shutil
import datetime
import argparse
import numpy as np
import skimage.color
import scipy.ndimage
from multiprocessing import Pool
from google.protobuf import text_format

#https://github.com/BVLC/caffe/issues/861#issuecomment-70124809
import matplotlib 
matplotlib.use('Agg')

parser = argparse.ArgumentParser(description="Extract neural network features for IMAGE input.")
parser.add_argument('--caffe_path', help='Path to caffe installation', required=False)
parser.add_argument('--img_filelist', help='Image file list.', required=True)
parser.add_argument('--img_root', help='Image file root dir.', default='/')
parser.add_argument('--prototxt', help='Model deploy file.', required=True)
parser.add_argument('--caffemodel', help='Model param file', required=False)
parser.add_argument('--feat_name', help='Feature name. e.g. fc7', required=True)
parser.add_argument('--lmdb', help='LMDB for saving the features.', required=True)
parser.add_argument('--gpu_index', help='GPU index (default=0).', type=int, default=0)
parser.add_argument('--pool_size', help='Pool size', type=int, default=8)
parser.add_argument('--mean_file', help='ImageNet mean file', required=False)
args = parser.parse_args()

if args.caffe_path:
    sys.path.append(os.path.join(args.caffe_path, 'python'))
import caffe
from caffe.proto import caffe_pb2

imagenet_mean = np.array([104, 117, 123])
if args.mean_file:
    imagenet_mean = np.load(args.mean_file)
    net_parameter = caffe_pb2.NetParameter()
    text_format.Merge(open(args.prototxt, 'r').read(), net_parameter)
    ratio = net_parameter.input_dim[2]*1.0/imagenet_mean.shape[1]
    imagenet_mean = scipy.ndimage.zoom(imagenet_mean, (1, ratio, ratio))

# INIT NETWORK
caffe.set_mode_gpu()
caffe.set_device(args.gpu_index)
net = caffe.Classifier(args.prototxt,args.caffemodel,
    mean=imagenet_mean,
    raw_scale=255,
    channel_swap=(2, 1, 0))

img_filenames = [args.img_root+'/'+x.rstrip() for x in open(args.img_filelist, 'r')]
N = len(img_filenames)

def array4d_idx_to_datum_string(array4d_idx):
    array4d = array4d_idx[0]
    idx = array4d_idx[1]
    global_idx = array4d_idx[2]
    array = array4d[idx, :, :, :]
    datum = caffe.io.array_to_datum(array.astype(float), global_idx)
    return datum.SerializeToString()      

if os.path.exists(args.lmdb):
    shutil.rmtree(args.lmdb)
env = lmdb.open(args.lmdb, map_size=int(1e12))

pool = Pool(args.pool_size)

## BATCH FORWARD 
BATCH_SIZE = int(net.blobs['data'].data.shape[0])
batch_num = int(math.ceil(N/float(BATCH_SIZE)))
print 'batch_num:', batch_num
for batch_idx in range(batch_num):
    start_idx = BATCH_SIZE * batch_idx
    end_idx = min(BATCH_SIZE * (batch_idx+1), N)
    print datetime.datetime.now().time(), '- batch: ', batch_idx, 'of', batch_num, 'idx range:[', start_idx, end_idx, ']'

    input_data = []
    for img_idx in range(start_idx, end_idx):
        im = caffe.io.load_image(img_filenames[img_idx])
        im = skimage.color.rgb2gray(im) 
        im = skimage.color.gray2rgb(im)
        input_data.append(im)

    net.predict(input_data, oversample=False)
    feats_blobproto = net.blobs[args.feat_name]
    feats_array = caffe.io.blobproto_to_array(feats_blobproto)
    array4d_idx = [(feats_array, idx, idx+start_idx) for idx in range(end_idx-start_idx)]
    datum_strings = pool.map(array4d_idx_to_datum_string, array4d_idx)
    with env.begin(write=True) as txn:
        for idx in range(end_idx-start_idx):
            txn.put('{:0>10d}'.format(start_idx+idx), datum_strings[idx])        

env.close();
