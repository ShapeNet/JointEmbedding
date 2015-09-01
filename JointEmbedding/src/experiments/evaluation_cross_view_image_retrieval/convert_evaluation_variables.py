#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(BASE_DIR)))
from global_variables import *

evaluation_variables_m_file = open('./evaluation_variables.m', 'w')

evaluation_folder = os.path.join(g_data_folder, 'evaluation_cross_view_image_retrieval')
evaluation_variables_m_file.write('evaluation_folder = \'%s\';\n' %(evaluation_folder))

evaluation_variables_m_file.close()
