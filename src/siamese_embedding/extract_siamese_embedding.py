#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import scipy.ndimage
from google.protobuf import text_format

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

siamese_embedding_caffemodel = os.path.join(g_siamese_embedding_testing_folder, 'snapshots%s_iter_%d.caffemodel'%(g_shapenet_synset_set_handle, args.iter_num))
siamese_embedding_prototxt = g_siamese_embedding_testing_prototxt

if args.caffemodel:
    siamese_embedding_caffemodel = args.caffemodel
if args.prototxt:
    siamese_embedding_prototxt = args.prototxt

print 'Siamese embedding for %s is:'%(args.image)
print siamese_embedding_array[0, :, 0, 0].tolist()

print 'Siamese embedding for %s is:'%(args.image)
siamese_embedding_array = extract_cnn_features(img_filelist=args.image,
                     img_root='/',
                     prototxt=siamese_embedding_prototxt, 
                     caffemodel=siamese_embedding_caffemodel,
                     feat_name='image_embedding',
                     caffe_path=g_caffe_install_path,
                     mean_file=g_mean_file)[0]
print siamese_embedding_array[0, :, 0, 0].tolist()
