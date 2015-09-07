#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
from subprocess import call

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from global_variables import *

if g_mirror_mode:
    print 'Error: mirror dataset can be created only when you are NOT in mirror mode! Toggle \'g_mirror_mode\' off in global_variables.py!'
    exit()
    
if len(g_mirror_name) == 0:
    print 'Error: mirror dataset can be created only when you have a non-empty mirror name! Edit \'g_mirror_name\' in global_variables.py!'
    exit()   
    
print 'Loading original shape list from %s...'%(g_shape_list_file)
shape_list_original = [line.strip().split(' ') for line in open(g_shape_list_file, 'r')]

shapenet_synset_set_handle_mirror = g_shapenet_synset_set_handle + '_' + g_mirror_name;
shape_list_file_mirror = os.path.join(g_data_folder, 'shape_list'+shapenet_synset_set_handle_mirror+'.txt')
print 'Loading original shape list from %s...'%(shape_list_file_mirror)
shape_list_mirror = [line.strip().split(' ') for line in open(shape_list_file_mirror, 'r')]

shape_list_mirror_mapping_filename = os.path.join(g_data_folder, 'shape_list_mapping'+shapenet_synset_set_handle_mirror+'.txt')
print 'Saving shape list mapping to %s...'%(shape_list_mirror_mapping_filename)
with open(shape_list_mirror_mapping_filename, 'w') as shape_list_mirror_mapping_file:
    for shape_item in shape_list_original:
        if shape_item in shape_list_mirror:
            shape_list_mirror_mapping_file.write('1\n')
        else:
            shape_list_mirror_mapping_file.write('0\n')
            
print 'Remember to toggle on \'g_mirror_mode\' in global_variables.py and play in mirror mode!'

call(['python', os.path.join(BASE_DIR, 'shape_embedding_training/mirror_shape_embedding_training.py')])
call(['python', os.path.join(BASE_DIR, 'image_embedding_training/mirror_image_embedding_training.py')])

