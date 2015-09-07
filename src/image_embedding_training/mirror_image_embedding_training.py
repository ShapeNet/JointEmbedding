#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

shapenet_synset_set_handle_mirror = g_shapenet_synset_set_handle + '_' + g_mirror_name;
filelist_mirror = os.path.join(g_data_folder, 'image_embedding/syn_images_filelist'+shapenet_synset_set_handle_mirror+'.txt')
filelist_mirror_train = os.path.join(g_data_folder, 'image_embedding/syn_images_filelist'+shapenet_synset_set_handle_mirror+'_train.txt')
filelist_mirror_val = os.path.join(g_data_folder, 'image_embedding/syn_images_filelist'+shapenet_synset_set_handle_mirror+'_val.txt')
imageid2shapeid_mirror = os.path.join(g_data_folder, 'image_embedding/syn_images_imageid2shapeid'+shapenet_synset_set_handle_mirror+'.txt')
imageid2shapeid_mirror_train = os.path.join(g_data_folder, 'image_embedding/syn_images_imageid2shapeid'+shapenet_synset_set_handle_mirror+'_train.txt')
imageid2shapeid_mirror_val = os.path.join(g_data_folder, 'image_embedding/syn_images_imageid2shapeid'+shapenet_synset_set_handle_mirror+'_val.txt')
train_val_split_mirror = os.path.join(g_data_folder, 'image_embedding/syn_images_train_val_split'+shapenet_synset_set_handle_mirror+'.txt')

shape_list_mirror_mapping_filename = os.path.join(g_data_folder, 'shape_list_mapping'+shapenet_synset_set_handle_mirror+'.txt')
print 'Loading shape list mapping from %s...'%(shape_list_mirror_mapping_filename)
shape_list_mirror_mapping = [bool(line.strip()) for line in open(shape_list_mirror_mapping_filename, 'r')]

print 'Loading original imageid2shapeid from %s...'%(g_syn_images_imageid2shapeid)
imageid2shapeid_original = [int(line.strip()) for line in open(g_syn_images_imageid2shapeid, 'r')]

print 'Loading original filelist from %s...'%(g_syn_images_filelist)
filelist_original = [line.strip() for line in open(g_syn_images_filelist, 'r')]

print 'Loading original train_val_split from %s...'%(g_syn_images_train_val_split)
train_val_split_original = [int(line.strip()) for line in open(g_syn_images_train_val_split, 'r')]

print 'Writing mirror files...'
filelist_mirror_file = open(filelist_mirror, 'w')
filelist_mirror_train_file = open(filelist_mirror_train, 'w')
filelist_mirror_val_file = open(filelist_mirror_val, 'w')
imageid2shapeid_mirror_file = open(imageid2shapeid_mirror, 'w')
imageid2shapeid_mirror_train_file = open(imageid2shapeid_mirror_train, 'w')
imageid2shapeid_mirror_val_file = open(imageid2shapeid_mirror_val, 'w')
train_val_split_mirror_file = open(train_val_split_mirror, 'w')

for idx, shapeid in enumerate(imageid2shapeid_original):
    if shape_list_mirror_mapping[shapeid]:
        filelist_mirror_file.write(filelist_original[idx]+'\n')
        imageid2shapeid_mirror_file.write(str(shapeid)+'\n')
        train_val = train_val_split_original[idx]
        train_val_split_mirror_file.write(str(int(train_val))+'\n')
        if train_val:
            filelist_mirror_train_file.write(filelist_original[idx]+'\n')
            imageid2shapeid_mirror_train_file.write(str(shapeid)+'\n')
        else:
            filelist_mirror_val_file.write(filelist_original[idx]+'\n')
            imageid2shapeid_mirror_val_file.write(str(shapeid)+'\n')          

filelist_mirror_file.close()
filelist_mirror_train_file.close()
filelist_mirror_val_file.close()
imageid2shapeid_mirror_file.close()
imageid2shapeid_mirror_train_file.close()
imageid2shapeid_mirror_val_file.close()
train_val_split_mirror_file.close()