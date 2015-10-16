#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import argparse
import fileinput

#https://github.com/BVLC/caffe/issues/861#issuecomment-70124809
import matplotlib 
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *
from utilities_caffe import stack_caffe_models

parser = argparse.ArgumentParser(description="Stitch pool5 extraction and image embedding caffemodels together.")
parser.add_argument('--iter_num', '-n', help='Use image embedding model trained after iter_num iterations', type=int, default=20000)
args = parser.parse_args()

image_embedding_testing_in = os.path.join(BASE_DIR, 'image_embedding_'+g_network_architecture_name+'.prototxt.in')
print 'Preparing %s...'%(g_image_embedding_testing_prototxt)
shutil.copy(image_embedding_testing_in, g_image_embedding_testing_prototxt)
for line in fileinput.input(g_image_embedding_testing_prototxt, inplace=True):
    line = line.replace('embedding_space_dim', str(g_shape_embedding_space_dimension))
    sys.stdout.write(line) 

image_embedding_caffemodel = os.path.join(g_image_embedding_training_folder, 'snapshots%s_iter_%d.caffemodel'%(g_shapenet_synset_set_handle, args.iter_num))
image_embedding_caffemodel_stacked = os.path.join(g_image_embedding_testing_folder, 'snapshots%s_iter_%d.caffemodel'%(g_shapenet_synset_set_handle, args.iter_num))

stack_caffe_models(prototxt=g_image_embedding_testing_prototxt,
                   base_model=g_fine_tune_caffemodel,
                   top_model=image_embedding_caffemodel,
                   stacked_model=image_embedding_caffemodel_stacked,
                   caffe_path=g_caffe_install_path)