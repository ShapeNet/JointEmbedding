#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from global_variables import *

folder_list = []
folder_list.append(g_lfd_images_folder)
folder_list.append(g_lfd_images_cropped_folder)
folder_list.append(g_syn_images_folder)
folder_list.append(g_syn_images_cropped_folder)
folder_list.append(g_syn_images_bkg_overlaid_folder)

shape_list = [line.strip().split(' ') for line in open(g_shape_list_file, 'r')]
print len(shape_list), 'Intermediate images of %d shapes are going to be deleted!'

for folder in folder_list:
    for item in shape_list:
        shape_folder = os.path.join(folder, item[0], item[1])
        if os.path.exists(shape_folder):
            shutil.rmtree(shape_folder)
