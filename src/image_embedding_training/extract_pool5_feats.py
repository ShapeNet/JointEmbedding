#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

script_path = os.path.join(BASE_DIR, 'extract_feature_batch.py')
cmd = 'python %s --caffe_path %s --img_filelist %s --prototxt %s --caffemodel %s --feat_name pool5 --lmdb %s --gpu_index %d --pool_size %d --mean_file %s' % \
(script_path, g_caffe_install_path, g_syn_images_filelist, g_extract_feat_prototxt, g_fine_tune_caffemodel, g_pool5_lmdb, g_extract_feat_gpu_index, g_extract_feat_thread_num, g_mean_file)

os.system(cmd)
