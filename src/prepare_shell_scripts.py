#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import stat
import shutil
import fileinput

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from global_variables import *

run_shape_embedding_training_sh_in = os.path.join(BASE_DIR, 'run_shape_embedding_training.sh.in')
run_shape_embedding_training_sh = os.path.join(BASE_DIR, 'run_shape_embedding_training.sh')
print 'Preparing %s...'%(run_shape_embedding_training_sh)
shutil.copy(run_shape_embedding_training_sh_in, run_shape_embedding_training_sh)
for line in fileinput.input(run_shape_embedding_training_sh, inplace=True):
    line = line.replace('/path/to/matlab', g_matlab_executable_path)
    sys.stdout.write(line)
st = os.stat(run_shape_embedding_training_sh)
os.chmod(run_shape_embedding_training_sh, st.st_mode | stat.S_IEXEC)
    
run_image_embedding_training_sh_in = os.path.join(BASE_DIR, 'run_image_embedding_training.sh.in')
run_image_embedding_training_sh = os.path.join(BASE_DIR, 'run_image_embedding_training.sh')
print 'Preparing %s...'%(run_image_embedding_training_sh)
shutil.copy(run_image_embedding_training_sh_in, run_image_embedding_training_sh)
for line in fileinput.input(run_image_embedding_training_sh, inplace=True):
    line = line.replace('/path/to/matlab', g_matlab_executable_path)
    sys.stdout.write(line)    
st = os.stat(run_shape_embedding_training_sh)
os.chmod(run_shape_embedding_training_sh, st.st_mode | stat.S_IEXEC)