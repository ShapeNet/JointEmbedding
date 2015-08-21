#!/usr/bin/python

import os
import sys
import stat
import shutil
import fileinput

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

if not os.path.exists(g_image_embedding_caffemodel):
    os.makedirs(g_image_embedding_caffemodel)
    
# Prepare train_val.prototxt
train_val_in = os.path.join(BASE_DIR, 'pool5_joint_embedding.prototxt.in')
train_val = os.path.join(g_image_embedding_caffemodel, 'train_val.prototxt')
print 'Preparing %s...'%(train_val)
shutil.copy(train_val_in, train_val)
for line in fileinput.input(train_val, inplace=True):
    line = line.replace('/path/to/syn_images_pool5_lmdb_train', g_pool5_lmdb_train)
    line = line.replace('/path/to/syn_images_pool5_lmdb_val', g_pool5_lmdb_val)
    line = line.replace('/path/to/shape_embedding_lmdb_train', g_shape_embedding_lmdb_train)
    line = line.replace('/path/to/shape_embedding_lmdb_val', g_shape_embedding_lmdb_val)
    line = line.replace('embedding_space_dim', str(g_shape_embedding_space_dimension))
    sys.stdout.write(line)
    
# Prepare train_val.prototxt
solver_in = os.path.join(BASE_DIR, 'solver.prototxt.in')
solver = os.path.join(g_image_embedding_caffemodel, 'solver.prototxt')
print 'Preparing %s...'%(solver)
shutil.copy(solver_in, solver)
for line in fileinput.input(solver, inplace=True):
    line = line.replace('_suffix', g_shapenet_synset_set_handle)
    sys.stdout.write(line)
    
# Prepare train_val.prototxt
command_in = os.path.join(BASE_DIR, 'run_training.sh.in')
command = os.path.join(g_image_embedding_caffemodel, 'run_training.sh')
print 'Preparing %s...'%(command)
shutil.copy(command_in, command)
for line in fileinput.input(command, inplace=True):
    line = line.replace('/path/to/caffe_executable', os.path.join(g_caffe_executable_path, 'bin/caffe'))
    line = line.replace('/path/to/caffemodel', g_fine_tune_caffemodel_file)
    sys.stdout.write(line)
st = os.stat(command)
os.chmod(command, st.st_mode | stat.S_IEXEC)