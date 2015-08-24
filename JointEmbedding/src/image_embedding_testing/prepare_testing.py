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

if not os.path.exists(g_image_embedding_testing_folder):
    os.makedirs(g_image_embedding_testing_folder)

parser = argparse.ArgumentParser(description="Stitch pool5 extraction and image embedding caffemodels together.")
parser.add_argument('--iter_num', '-n', help='Use image embedding model trained after iter_num iterations', type=int, default=20000)
args = parser.parse_args()

image_embedding_testing_in = os.path.join(BASE_DIR, 'image_embedding.prototxt.in')
print 'Preparing %s...'%(g_image_embedding_testing_prototxt)
shutil.copy(image_embedding_testing_in, g_image_embedding_testing_prototxt)
for line in fileinput.input(g_image_embedding_testing_prototxt, inplace=True):
    line = line.replace('embedding_space_dim', str(g_shape_embedding_space_dimension))
    sys.stdout.write(line) 
net = caffe.Net(g_image_embedding_testing_prototxt, caffe.TEST)

print 'Copying trained layers from %s...'%(g_fine_tune_caffemodel)
net.copy_from(g_fine_tune_caffemodel)

image_embedding_caffemodel = os.path.join(g_image_embedding_training_folder, 'snapshots%s_iter_%d.caffemodel'%(g_shapenet_synset_set_handle, args.iter_num))
print 'Copying trained layers from %s...'%(image_embedding_caffemodel)
net.copy_from(image_embedding_caffemodel)

image_embedding_caffemodel = os.path.join(g_image_embedding_testing_folder, 'snapshots%s_iter_%d.caffemodel'%(g_shapenet_synset_set_handle, args.iter_num))
net.save(image_embedding_caffemodel)