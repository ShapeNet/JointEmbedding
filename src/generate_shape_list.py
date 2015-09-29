#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from global_variables import *

print 'Generating shape list...'

blacklist = [line.strip().split(' ') for line in open('./blacklist.txt', 'r')]

shape_list = []
for synset in g_shapenet_synset_set:
    synset_folder = os.path.join(g_shapenet_root_folder, synset)
    
    shape_folders = os.listdir(synset_folder)
    for md5 in shape_folders:
        shape_file = os.path.join(synset_folder, md5, 'model.obj')
        
        if not os.path.exists(shape_file):
            print shape_file, 'doesn\'t exist!'
        
        shape_item = [synset, md5]
	if shape_item in blacklist:
            print 'Skip %s/%s as it\'s in the blacklist' % (synset, md5)
        else:
            shape_list.append([synset, md5])
        
print len(shape_list), 'shapes are collected!'

shape_list.sort()
shape_list_file = open(g_shape_list_file, 'w')
for shape_property in shape_list:
    shape_list_file.write(shape_property[0]+' '+shape_property[1]+'\n');
shape_list_file.close()
