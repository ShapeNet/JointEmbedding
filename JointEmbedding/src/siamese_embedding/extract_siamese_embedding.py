#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import argparse

#https://github.com/BVLC/caffe/issues/861#issuecomment-70124809
import matplotlib 
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

parser = argparse.ArgumentParser(description="Extract siamese embedding features for IMAGE input.")
parser.add_argument('--image', help='Path to input image (cropped)', required=False)
parser.add_argument('--iter_num', '-n', help='Use caffemodel trained after iter_num iterations', type=int, default=20000)
parser.add_argument('--caffemodel', '-c', help='Path to caffemodel (will ignore -n option if provided)', required=False)
parser.add_argument('--prototxt', '-p', help='Path to prototxt (if not at the default place)', required=False)
parser.add_argument('--gpu_index', help='GPU index (default=0).', type=int, default=0)
args = parser.parse_args()

sys.path.append(os.path.join(g_caffe_install_path, 'python'))
import caffe

caffemodel = os.path.join(g_siamese_embedding_testing_folder, 'snapshots%s_iter_%d.caffemodel'%(g_shapenet_synset_set_handle, args.iter_num))
prototxt = g_siamese_embedding_testing_prototxt

if args.caffemodel:
    caffemodel = args.caffemodel
if args.prototxt:
    prototxt = args.prototxt

# INIT NETWORK
caffe.set_mode_gpu()
caffe.set_device(args.gpu_index)
net = caffe.Classifier(prototxt,
                       caffemodel,
                       mean=np.array([104, 117, 123]),
                       raw_scale=255,
                       channel_swap=(2, 1, 0))

input_data = []
im = caffe.io.load_image(args.image)
input_data.append(im)

net.predict(input_data, oversample=False)
siamese_embedding_blobproto = net.blobs['image_embedding']
siamese_embedding_array = caffe.io.blobproto_to_array(siamese_embedding_blobproto)

print 'Siamese embedding for %s is:'%(args.image)
print siamese_embedding_array[0, :, 0, 0].tolist()
