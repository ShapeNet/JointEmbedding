#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
from subprocess import call

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *
from utilities_common import query_yes_no

if not os.path.exists(g_sun2012_data_folder):
    os.mkdir(g_sun2012_data_folder) 

# download and unzip SUN2012 data
filename = os.path.join(g_sun2012_data_folder, '../SUN2012pascalformat.tar.gz')
if os.path.exists(filename):
    if not query_yes_no(('It seems that you have downloaded SUN2012 data to \"%s\", skip downloading?')%(filename)):
        os.remove(filename)
        call(['wget', '-O', filename, g_sun2012_data_url])

call(['tar', 'xvzf', filename, '-C', g_sun2012_data_folder, '--strip-components=1'])
filelist = open(g_syn_bkg_filelist, 'w')
for filename in sorted(os.listdir(g_syn_bkg_folder)):
    filelist.write(filename+'\n')
filelist.close()
#os.remove(filename)
