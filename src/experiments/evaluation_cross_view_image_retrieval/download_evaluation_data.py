#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
from subprocess import call

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(BASE_DIR)))
from global_variables import *
from utilities_common import query_yes_no

# download and unzip evaluation data
url = 'https://shapenet.cs.stanford.edu/projects/JointEmbedding/data/evaluation_cross_view_image_retrieval.zip'
filename = os.path.join(g_data_folder, 'evaluation_cross_view_image_retrieval.zip')

if os.path.exists(filename):
    if query_yes_no(('It seems that you have downloaded evaluation_cross_view_image_retrieval data to \"%s\", skip downloading?')%(filename)):
        exit()
    else:
        os.remove(filename)
        
call(['wget', '-O', filename, url])
call(['unzip', filename, '-d', g_data_folder])
