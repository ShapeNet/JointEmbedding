#!/usr/bin/python

import os
import sys
import stat
import random
import shutil
import fileinput

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

##############################################################################
# Sample image_image pairs
##############################################################################
print 'Loading imageid2shapeid from %s...'%(g_syn_images_imageid2shapeid)
imageid2shapeid = [int(line.strip()) for line in open(g_syn_images_imageid2shapeid, 'r')]    

print 'Loading shape distance matrix from %s...'%(g_shape_distance_matrix_file_txt)
distance_matrix = [[float(value) for value in line.strip().split(' ')] for line in open(g_shape_distance_matrix_file_txt, 'r')]

print 'Computing distance ranking...'
distance_ranking = []
for distances in distance_matrix:
    ranking = range(len(distances))
    ranking.sort(key = lambda rank:distances[rank])
    distance_ranking.append(ranking)

shape_num = len(distance_matrix)
image_num = len(imageid2shapeid)

print 'Building model id to image id mapping...'
model_id_to_image_ids = [[]]*shape_num
for image_id, shape_id in enumerate(imageid2shapeid):
    model_id_to_image_ids[shape_id].append(image_id)

average_distance = 0.0
image_image_pairs = []
random.seed(9527) # seed random with a fixed number
for idx in range(g_siamese_pairs_num):
    if idx%100000 == 0:
        print 'Processing', idx, 'of', g_siamese_pairs_num, '...'

    is_positive_sample = (random.randint(0,2099) < 100) # positive:negative = 1:20
    image_id_1 = random.randint(0, image_num-1)
    shape_id_1 = imageid2shapeid[image_id_1]
    
    if is_positive_sample:
        idx_shape_id_2 = min(int(abs(random.gauss(0, g_siamese_pair_top_k/3))), g_siamese_pair_top_k)
    else:
        idx_shape_id_2 = random.randint(g_siamese_pair_top_k+1, shape_num-1)
        
    shape_id_2 = distance_ranking[shape_id_1][idx_shape_id_2]
    image_id_2 = model_id_to_image_ids[shape_id_2][random.randint(0, len(model_id_to_image_ids[shape_id_2])-1)]
    image_image_pairs.append((image_id_1, image_id_2, (int)(is_positive_sample)))
    
    shape_id_top_k = distance_ranking[shape_id_1][max(g_siamese_pair_top_k, 32)]
    average_distance = average_distance + distance_matrix[shape_id_1][shape_id_top_k]
    
average_distance = average_distance/g_siamese_pairs_num
print 'Average distance: ', average_distance, '!'

with open(g_syn_images_pairs_filelist, 'w') as image_image_pairs_file:
    for pair in image_image_pairs:
        image_image_pairs_file.write(str(pair[0])+' '+str(pair[1])+' '+str(pair[2])+'\n')
    
# Prepare train_val.prototxt
train_val_in = os.path.join(BASE_DIR, 'train_val_'+g_network_architecture_name+'.prototxt.in')
print 'Preparing %s...'%(g_siamese_embedding_train_val_prototxt)
shutil.copy(train_val_in, g_siamese_embedding_train_val_prototxt)
for line in fileinput.input(g_siamese_embedding_train_val_prototxt, inplace=True):
    line = line.replace('/path/to/syn_images_pairs_pool5_lmdb_train', g_pairs_pool5_lmdb_train)
    line = line.replace('/path/to/syn_images_pairs_pool5_lmdb_val', g_pairs_pool5_lmdb_val)
    line = line.replace('embedding_space_dim', str(g_shape_embedding_space_dimension))
    line = line.replace('contrastive_loss_margin', str(average_distance))
    sys.stdout.write(line)
    
# Prepare train_val.prototxt
solver_in = os.path.join(BASE_DIR, 'solver.prototxt.in')
print 'Preparing %s...'%(g_siamese_embedding_solver_prototxt)
shutil.copy(solver_in, g_siamese_embedding_solver_prototxt)
for line in fileinput.input(g_siamese_embedding_solver_prototxt, inplace=True):
    line = line.replace('NETWORK_ARCHITECTURE_NAME', g_network_architecture_name)
    line = line.replace('_SUFFIX', g_shapenet_synset_set_handle)
    sys.stdout.write(line)
    
# Prepare train_val.prototxt
command_in = os.path.join(BASE_DIR, 'run_training.sh.in')
print 'Preparing %s...'%(g_siamese_embedding_command_sh)
shutil.copy(command_in, g_siamese_embedding_command_sh)
for line in fileinput.input(g_siamese_embedding_command_sh, inplace=True):
    line = line.replace('/path/to/caffe_executable', os.path.join(g_caffe_install_path, 'bin/caffe'))
    line = line.replace('/path/to/caffemodel', g_fine_tune_caffemodel)
    sys.stdout.write(line)
st = os.stat(g_siamese_embedding_command_sh)
os.chmod(g_siamese_embedding_command_sh, st.st_mode | stat.S_IEXEC)
