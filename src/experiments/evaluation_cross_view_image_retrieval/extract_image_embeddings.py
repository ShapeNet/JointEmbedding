#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import argparse
import datetime
import fileinput
import numpy as np
from google.protobuf import text_format

#https://github.com/BVLC/caffe/issues/861#issuecomment-70124809
import matplotlib 
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(BASE_DIR)))
from global_variables import *

parser = argparse.ArgumentParser(description="Extract image embedding features for IMAGE input.")
parser.add_argument('--dataset', '-d', help='The dataset for extracting image embedding features(02958343, 03001627_clutter, or 03001627_clean)', required=True)
parser.add_argument('--iter_num', '-n', help='Use caffemodel trained after iter_num iterations', type=int, default=20000)
parser.add_argument('--caffemodel', '-c', help='Path to caffemodel (will ignore -n option if provided)', required=False)
parser.add_argument('--prototxt', '-p', help='Path to prototxt (if not at the default place)', required=False)
parser.add_argument('--gpu_index', help='GPU index (default=0).', type=int, default=0)
args = parser.parse_args()

sys.path.append(os.path.join(g_caffe_install_path, 'python'))
import caffe
from caffe.proto import caffe_pb2

dataset_handle = '_'+args.dataset.split('_')[0]   
caffemodel = os.path.join(g_image_embedding_testing_folder, 'snapshots%s_iter_%d.caffemodel'%(dataset_handle, args.iter_num))
prototxt = g_image_embedding_testing_prototxt

evaluation_folder = os.path.join(g_data_folder, 'evaluation_cross_view_image_retrieval')
    
if args.caffemodel:
    caffemodel = args.caffemodel
    
if args.prototxt:
    prototxt = args.prototxt
else:
    # change the batch size to speedup the computation a bit
    evaluation_prototxt = os.path.join(evaluation_folder, os.path.split(prototxt)[-1])
    print 'Preparing %s...'%(evaluation_prototxt)
    shutil.copy(prototxt, evaluation_prototxt)
    for line in fileinput.input(evaluation_prototxt, inplace=True):
        line = line.replace('dim: 1', 'dim: 256')
        sys.stdout.write(line) 
    prototxt = evaluation_prototxt
    
net_parameter = caffe_pb2.NetParameter()
text_format.Merge(open(prototxt, 'r').read(), net_parameter)
input_shape = net_parameter.input_shape[0].dim
batch_size = input_shape[0]

# INIT NETWORK
caffe.set_mode_gpu()
caffe.set_device(args.gpu_index)
net = caffe.Classifier(prototxt,
                       caffemodel,
                       mean=np.array([104, 117, 123]),
                       raw_scale=255,
                       channel_swap=(2, 1, 0))

input_data = [None]*batch_size
filelist = [line.strip() for line in open(os.path.join(evaluation_folder, 'filelist_'+args.dataset+'.txt'))]
print datetime.datetime.now().time(), 'Computing image embedding for %s (%d images)...'%(args.dataset, len(filelist)) 

data_folder = os.path.join(evaluation_folder, 'images_'+args.dataset)
image_embedding_filename = os.path.join(evaluation_folder, 'image_embeddings_'+args.dataset+'.txt')
with open(image_embedding_filename, 'w') as image_embedding_file:
    for idx, filename in enumerate(filelist):
        input_data[idx%batch_size] = caffe.io.load_image(os.path.join(data_folder, filename))
        if idx%batch_size == batch_size -1 or idx == len(filelist)-1:
            net.predict(input_data, oversample=False)
            image_embedding_blobproto = net.blobs['image_embedding']
            image_embedding_array = caffe.io.blobproto_to_array(image_embedding_blobproto)
    
            image_embedding_num = batch_size
            if idx == len(filelist)-1:
                image_embedding_num = idx%batch_size + 1
            print datetime.datetime.now().time(), 'Saving image embedding for images [%d-%d]...'%(idx+1-batch_size, idx)
            for i in range(image_embedding_num):
                image_embedding = image_embedding_array[i, :].reshape(g_shape_embedding_space_dimension).tolist()
                image_embedding_file.write(' '.join([str(value) for value in image_embedding])+'\n')
    
