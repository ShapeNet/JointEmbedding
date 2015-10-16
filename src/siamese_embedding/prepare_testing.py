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

sys.path.append(os.path.join(g_caffe_install_path, 'python'))
import caffe

parser = argparse.ArgumentParser(description="Stitch pool5 extraction and siamese embedding caffemodels together.")
parser.add_argument('--iter_num', '-n', help='Use siamese embedding model trained after iter_num iterations', type=int, default=20000)
args = parser.parse_args()

siamese_embedding_testing_in = os.path.join(BASE_DIR, 'siamese_embedding_'+g_network_architecture_name+'.prototxt.in')
print 'Preparing %s...'%(g_siamese_embedding_testing_prototxt)
shutil.copy(siamese_embedding_testing_in, g_siamese_embedding_testing_prototxt)
for line in fileinput.input(g_siamese_embedding_testing_prototxt, inplace=True):
    line = line.replace('embedding_space_dim', str(g_shape_embedding_space_dimension))
    sys.stdout.write(line) 

siamese_embedding_caffemodel = os.path.join(g_siamese_embedding_training_folder, 'snapshots%s_iter_%d.caffemodel'%(g_shapenet_synset_set_handle, args.iter_num))
siamese_embedding_caffemodel_stacked = os.path.join(g_siamese_embedding_testing_folder, 'snapshots%s_iter_%d.caffemodel'%(g_shapenet_synset_set_handle, args.iter_num))

stack_caffe_models(prototxt=g_siamese_embedding_testing_prototxt,
                   base_model=g_fine_tune_caffemodel,
                   top_model=siamese_embedding_caffemodel,
                   stacked_model=siamese_embedding_caffemodel_stacked,
                   caffe_path=g_caffe_install_path)
