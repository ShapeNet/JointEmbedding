#!/usr/bin/python

import os
import sys
import stat
import shutil
import fileinput

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

if not os.path.exists(g_image_embedding_training_folder):
    os.makedirs(g_image_embedding_training_folder)
    
# Prepare train_val.prototxt
train_val_in = os.path.join(BASE_DIR, 'train_val_'+g_network_architecture_name+'.prototxt.in')
print 'Preparing %s...'%(g_image_embedding_train_val_prototxt)
shutil.copy(train_val_in, g_image_embedding_train_val_prototxt)
for line in fileinput.input(g_image_embedding_train_val_prototxt, inplace=True):
    line = line.replace('/path/to/syn_images_pool5_lmdb_train', g_pool5_lmdb_train)
    line = line.replace('/path/to/syn_images_pool5_lmdb_val', g_pool5_lmdb_val)
    line = line.replace('/path/to/shape_embedding_lmdb_train', g_shape_embedding_lmdb_train)
    line = line.replace('/path/to/shape_embedding_lmdb_val', g_shape_embedding_lmdb_val)
    line = line.replace('embedding_space_dim', str(g_shape_embedding_space_dimension))
    sys.stdout.write(line)
    
# Prepare train_val.prototxt
solver_in = os.path.join(BASE_DIR, 'solver.prototxt.in')
print 'Preparing %s...'%(g_image_embedding_solver_prototxt)
shutil.copy(solver_in, g_image_embedding_solver_prototxt)
for line in fileinput.input(g_image_embedding_solver_prototxt, inplace=True):
    line = line.replace('NETWORK_ARCHITECTURE_NAME', g_network_architecture_name)
    line = line.replace('_SUFFIX', g_shapenet_synset_set_handle)
    sys.stdout.write(line)
    
# Prepare train_val.prototxt
command_in = os.path.join(BASE_DIR, 'run_training.sh.in')
print 'Preparing %s...'%(g_image_embedding_command_sh)
shutil.copy(command_in, g_image_embedding_command_sh)
for line in fileinput.input(g_image_embedding_command_sh, inplace=True):
    line = line.replace('/path/to/caffe_executable', os.path.join(g_caffe_install_path, 'bin/caffe'))
    line = line.replace('/path/to/caffemodel', g_fine_tune_caffemodel)
    sys.stdout.write(line)
st = os.stat(g_image_embedding_command_sh)
os.chmod(g_image_embedding_command_sh, st.st_mode | stat.S_IEXEC)
