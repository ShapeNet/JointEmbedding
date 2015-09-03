#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

filelist = [line.strip() for line in open(g_syn_images_filelist, 'r')]
imageid2shapeid = [line.strip() for line in open(g_syn_images_imageid2shapeid, 'r')]
image_num = len(filelist)
train_val_split = [1]*image_num
val_num = int(image_num*(1-g_train_ratio))
train_val_split[0:val_num] = [0]*val_num

random.seed(9527) # seed random with a fixed number
random.shuffle(train_val_split)

filelist_train = open(g_syn_images_filelist_train, 'w')
filelist_val = open(g_syn_images_filelist_val, 'w')
imageid2shapeid_train = open(g_syn_images_imageid2shapeid_train, 'w')
imageid2shapeid_val = open(g_syn_images_imageid2shapeid_val, 'w')
train_val_split_file = open(g_syn_images_train_val_split, 'w')

for idx, train_val in enumerate(train_val_split):
    if train_val:
        filelist_train.write(filelist[idx]+'\n')
        imageid2shapeid_train.write(imageid2shapeid[idx]+'\n')
    else:
        filelist_val.write(filelist[idx]+'\n')
        imageid2shapeid_val.write(imageid2shapeid[idx]+'\n')
    train_val_split_file.write(str(train_val)+'\n')

filelist_train.close()
filelist_val.close()
imageid2shapeid_train.close()
imageid2shapeid_val.close()
train_val_split_file.close()
