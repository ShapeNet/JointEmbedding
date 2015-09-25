#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import argparse
from google.protobuf import text_format

#https://github.com/BVLC/caffe/issues/861#issuecomment-70124809
import matplotlib 
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

parser = argparse.ArgumentParser(description="Extract image embedding features for IMAGE input.")
parser.add_argument('--image', help='Path to input image (cropped)', required=False)
parser.add_argument('--iter_num', '-n', help='Use caffemodel trained after iter_num iterations', type=int, default=20000)
parser.add_argument('--caffemodel', '-c', help='Path to caffemodel (will ignore -n option if provided)', required=False)
parser.add_argument('--prototxt', '-p', help='Path to prototxt (if not at the default place)', required=False)
parser.add_argument('--gpu_index', help='GPU index (default=0).', type=int, default=0)
args = parser.parse_args()

sys.path.append(os.path.join(g_caffe_install_path, 'python'))
import caffe
from caffe.proto import caffe_pb2

caffemodel = os.path.join(g_image_embedding_testing_folder, 'snapshots%s_iter_%d.caffemodel'%(g_shapenet_synset_set_handle, args.iter_num))
prototxt = g_image_embedding_testing_prototxt

if args.caffemodel:
    caffemodel = args.caffemodel
if args.prototxt:
    prototxt = args.prototxt
    
imagenet_mean = np.load(g_mean_file)
net_parameter = caffe_pb2.NetParameter()
text_format.Merge(open(prototxt, 'r').read(), net_parameter)
input_shape = net_parameter.input_shape[0].dim
ratio = input_shape[2]*1.0/imagenet_mean.shape[1]
imagenet_mean = caffe.io.resize_image(imagenet_mean.transpose((1, 2, 0)), net_parameter.input_dim[2:]).transpose((2, 0, 1))

# INIT NETWORK
caffe.set_mode_gpu()
caffe.set_device(args.gpu_index)
net = caffe.Classifier(prototxt,
                       caffemodel,
                       #mean=np.array([104, 117, 123]),
                       mean=imagenet_mean,
                       raw_scale=255,
                       channel_swap=(2, 1, 0))

input_data = []
im = caffe.io.load_image(args.image)
input_data.append(im)

net.predict(input_data, oversample=False)
image_embedding_blobproto = net.blobs['image_embedding']
image_embedding_array = caffe.io.blobproto_to_array(image_embedding_blobproto)

print 'Image embedding for %s is:'%(args.image)
print image_embedding_array[0, :, 0, 0].tolist()
