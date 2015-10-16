#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

shape_list = [line.strip().split(' ') for line in open(g_shape_list_file,'r')]

image_list_train = [line.strip() for line in open(g_syn_images_filelist_train, 'r')]
imageid2shapeid_train = [int(line.strip()) for line in open(g_syn_images_imageid2shapeid_train, 'r')]
for idx, shapeid in enumerate(imageid2shapeid_train):
    if (shape_list[shapeid][0] not in image_list_train[idx]) or (shape_list[shapeid][1] not in image_list_train[idx]):
        print idx, shapeid, shape_list[shapeid], image_list_train[idx]
    assert((shape_list[shapeid][0] in image_list_train[idx]) and (shape_list[shapeid][1] in image_list_train[idx]))

image_list_val = [line.strip() for line in open(g_syn_images_filelist_val, 'r')]
imageid2shapeid_val = [int(line.strip()) for line in open(g_syn_images_imageid2shapeid_val, 'r')]
for idx, shapeid in enumerate(imageid2shapeid_val):
    if (shape_list[shapeid][0] not in image_list_val[idx]) or (shape_list[shapeid][1] not in image_list_val[idx]):
        print idx, shapeid, shape_list[shapeid], image_list_val[idx]
    assert((shape_list[shapeid][0] in image_list_val[idx]) and (shape_list[shapeid][1] in image_list_val[idx]))
