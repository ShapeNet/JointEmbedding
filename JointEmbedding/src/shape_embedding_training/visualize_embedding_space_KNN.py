#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import argparse
import fileinput
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

parser = argparse.ArgumentParser(description="Visualize the top K nearest neighbor of the query_idx-th shape.")
parser.add_argument('--query_idx', help='Query shape index.', type=int, required=True)
parser.add_argument('--top_k', help='Retrieve top K shapes.', type=int, default=32)
args = parser.parse_args()

print 'Loading shape embedding space from %s...'%(g_shape_embedding_space_file_txt)
shape_embedding_space = [np.array([float(value) for value in line.strip().split(' ')]) for line in open(g_shape_embedding_space_file_txt, 'r')]
query_embedding = shape_embedding_space[args.query_idx]

print 'Computing distances and ranking...'
sorted_distances = sorted([(sum((query_embedding-shape_embedding)**2), idx) for idx, shape_embedding in enumerate(shape_embedding_space)])

print 'Loading shape list from %s'%(g_shape_list_file)
shape_list = [line.strip().split(' ') for line in open(g_shape_list_file, 'r')]
assert(len(shape_embedding_space) == len(shape_list))

visualization_folder = os.path.join(BASE_DIR, 'visualization')
if not os.path.exists(visualization_folder):
    os.makedirs(visualization_folder)
visualization_filename = os.path.join(visualization_folder, 'visualize_embedding_space_KNN_%s.html'%(args.query_idx))
visualization_template = os.path.join(BASE_DIR, 'visualize_embedding_space_KNN.html')
print 'Saving visualization to %s...'%(visualization_filename)
shutil.copy(visualization_template, visualization_filename)
for line in fileinput.input(visualization_filename, inplace=True):
    line = line.replace('SYNSET', shape_list[args.query_idx][0])
    line = line.replace('MD5_ID', shape_list[args.query_idx][1])
    sys.stdout.write(line)

retrieval_list = ''
for i in range(args.top_k):
    shape_idx = sorted_distances[i][1]
    synset = shape_list[shape_idx][0]
    md5_id = shape_list[shape_idx][1]
    retrieval_list = retrieval_list + \
"""
     <div class="retrieval">
	    <span class="helper"></span>
	    <img class="item" src="https://shapenet.cs.stanford.edu/shapenet_brain/media/shape_lfd_images/%s/%s/%s_%s_a054_e020_t000_d003.png" title="%s/%s">
	    <div class="property">
		<p>id: %s</p>
	    </div>
	</div>
 """%(synset, md5_id, synset, md5_id, synset, md5_id, md5_id)
 

for line in fileinput.input(visualization_filename, inplace=True):
    line = line.replace('RETRIEVAL_LIST', retrieval_list)
    sys.stdout.write(line)
