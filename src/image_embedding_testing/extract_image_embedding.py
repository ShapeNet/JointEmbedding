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
from utilities_caffe import *

parser = argparse.ArgumentParser(description="Extract image embedding features for IMAGE input.")
parser.add_argument('--image', help='Path to input image (cropped)', required=True)
parser.add_argument('--iter_num', '-n', help='Use caffemodel trained after iter_num iterations', type=int, default=20000)
parser.add_argument('--caffemodel', '-c', help='Path to caffemodel (will ignore -n option if provided)', required=False)
parser.add_argument('--prototxt', '-p', help='Path to prototxt (if not at the default place)', required=False)
parser.add_argument('--gpu_index', help='GPU index (default=0).', type=int, default=0)
args = parser.parse_args()

image_embedding_caffemodel = os.path.join(g_image_embedding_testing_folder, 'snapshots%s_iter_%d.caffemodel'%(g_shapenet_synset_set_handle, args.iter_num))
image_embedding_prototxt = g_image_embedding_testing_prototxt

if args.caffemodel:
    image_embedding_caffemodel = args.caffemodel
if args.prototxt:
    image_embedding_prototxt = args.prototxt

print 'Image embedding for %s is:'%(args.image)
image_embedding_array = extract_cnn_features(img_filelist=args.image,
                     img_root='/',
                     prototxt=image_embedding_prototxt, 
                     caffemodel=image_embedding_caffemodel,
                     feat_name='image_embedding',
                     caffe_path=g_caffe_install_path,
                     mean_file=g_mean_file)[0]
print image_embedding_array.tolist()
