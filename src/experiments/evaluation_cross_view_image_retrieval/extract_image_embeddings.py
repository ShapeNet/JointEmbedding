#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import argparse
import datetime
import fileinput

#https://github.com/BVLC/caffe/issues/861#issuecomment-70124809
import matplotlib 
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(BASE_DIR)))
from global_variables import *
from utilities_caffe import *

parser = argparse.ArgumentParser(description="Extract image embedding features for IMAGE input.")
parser.add_argument('--dataset', '-d', help='The dataset for extracting image embedding features(02958343, 03001627_clutter, or 03001627_clean)', required=True)
parser.add_argument('--iter_num', '-n', help='Use caffemodel trained after iter_num iterations', type=int, default=20000)
parser.add_argument('--caffemodel', '-c', help='Path to caffemodel (will ignore -n option if provided)', required=False)
parser.add_argument('--prototxt', '-p', help='Path to prototxt (if not at the default place)', required=False)
parser.add_argument('--gpu_index', help='GPU index (default=0).', type=int, default=0)
args = parser.parse_args()

dataset_handle = '_'+args.dataset.split('_')[0]   
image_embedding_caffemodel = os.path.join(g_image_embedding_testing_folder, 'snapshots%s_iter_%d.caffemodel'%(dataset_handle, args.iter_num))
image_embedding_prototxt = g_image_embedding_testing_prototxt

evaluation_folder = os.path.join(g_data_folder, 'evaluation_cross_view_image_retrieval')
    
if args.caffemodel:
    image_embedding_caffemodel = args.caffemodel
    
if args.prototxt:
    image_embedding_prototxt = args.prototxt
else:
    # change the batch size to speedup the computation a bit
    evaluation_prototxt = os.path.join(evaluation_folder, os.path.split(image_embedding_prototxt)[-1])
    print 'Preparing %s...'%(evaluation_prototxt)
    shutil.copy(image_embedding_prototxt, evaluation_prototxt)
    for line in fileinput.input(evaluation_prototxt, inplace=True):
        sys.stdout.write(line.replace('dim: 1', 'dim: 256')) 
    image_embedding_prototxt = evaluation_prototxt

filelist_filename = os.path.join(evaluation_folder, 'filelist_'+args.dataset+'.txt')
data_folder = os.path.join(evaluation_folder, 'images_'+args.dataset)
image_embedding_filename = os.path.join(evaluation_folder, 'image_embeddings_'+args.dataset+'.txt')
extract_cnn_features(img_filelist=filelist_filename,
                     img_root=data_folder,
                     prototxt=image_embedding_prototxt, 
                     caffemodel=image_embedding_caffemodel,
                     feat_name='image_embedding',
                     output_path=image_embedding_filename,
                     output_type='txt',
                     caffe_path=g_caffe_install_path,
                     mean_file=g_mean_file)
