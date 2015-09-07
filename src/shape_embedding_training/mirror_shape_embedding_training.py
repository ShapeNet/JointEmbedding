#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

shapenet_synset_set_handle_mirror = g_shapenet_synset_set_handle + '_' + g_mirror_name;
shape_list_mirror_mapping_filename = os.path.join(g_data_folder, 'shape_list_mapping'+shapenet_synset_set_handle_mirror+'.txt')
print 'Loading shape list mapping from %s...'%(shape_list_mirror_mapping_filename)
shape_list_mirror_mapping = [bool(int(line.strip())) for line in open(shape_list_mirror_mapping_filename, 'r')]

print 'Loading original shape embedding space from %s...'%(g_shape_embedding_space_file_txt)
embedding_space = [line.strip() for line in open(g_shape_embedding_space_file_txt, 'r')]

shape_embedding_space_file_txt_mirror = os.path.join(g_data_folder, 'shape_embedding/shape_embedding_space'+shapenet_synset_set_handle_mirror+'.txt')
print 'Saving mirror shape embedding space to %s...'%(shape_embedding_space_file_txt_mirror)
with open(shape_embedding_space_file_txt_mirror, 'w') as shape_embedding_space_file_txt_mirror_file:
    for idx, mirror in enumerate(shape_list_mirror_mapping):
        if mirror:
            shape_embedding_space_file_txt_mirror_file.write(embedding_space[idx]+'\n')
