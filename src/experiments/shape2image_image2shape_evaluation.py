import numpy as np
import math
import argparse
import subprocess
import tempfile
from scipy.spatial.distance import *
import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))
EXACTMATCH_DATASET = os.path.join(BASEDIR, 'ExactMatchChairsDataset')
RESULTDIR = os.path.join(BASEDIR,'results')

parser = argparse.ArgumentParser(description="shape2image and image2shape evaluation on exact-match-chair dataset.")
parser.add_argument('-m', '--img_model_ids_file', help='File, each line is model id (0to6776) of img.', required=True)
parser.add_argument('-i', '--input_npy_file', help='Input numpy file (pool5 feature), contains an array of N*C*H*W', required=True)
parser.add_argument('-s', '--shape_embedding_file', help='Shape embedding txt file (#model * #embedding-dim).', required=False)
parser.add_argument('-d', '--model_deploy_file', help='Caffe model deploy file (batch size = 50).', required=False)
parser.add_argument('-p', '--model_param_file', help='Caffe model parameter file.', required=False)
parser.add_argument('-n1', '--nb_image2shape', help='Number of nearest shapes', required=True)
parser.add_argument('-n2', '--nb_shape2image', help='Number of nearest images', required=True)
parser.add_argument('--result_id', help='Result ID (string)', required=True)
parser.add_argument('--distance_matrix_txt_file', help='Distance matrix (#image * #model) txt file (default=None)', default=None, required=False)
parser.add_argument('--clutter_only', help='Test on clutter image only.', action='store_true')
parser.add_argument('--clean_only', help='Test on clean image only.', action='store_true')
parser.add_argument('--feat_dim', help='Embedding feat dim (default=100)', default=100, required=False)

args = parser.parse_args()

# 315 test image names
img_names = [x.rstrip() for x in open(os.path.join(EXACTMATCH_DATASET, 'exact_match_chairs_img_filelist.txt'),'r')]
# 105 modelIds of exact match dataset
exact_match_modelIds = [int(x.rstrip()) for x in open(os.path.join(EXACTMATCH_DATASET, 'exact_match_chairs_shape_modelIds_0to6776.txt'),'r')]
# 141 cluttered image index (in 315 test images) 
exact_match_cluttered_indicies = [int(x.rstrip()) for x in open(os.path.join(EXACTMATCH_DATASET, 'exact_match_chairs_cluttered_img_indicies_0to314.txt'),'r')]
exact_match_clean_indicies = [x for x in range(315) if x not in exact_match_cluttered_indicies]
# 315 image model ids
image_model_ids = np.loadtxt(args.img_model_ids_file)

if args.clutter_only:
  image_model_ids = image_model_ids[exact_match_cluttered_indicies]
  img_names = [img_names[k] for k in exact_match_cluttered_indicies]
elif args.clean_only:
  image_model_ids = image_model_ids[exact_match_clean_indicies]
  img_names = [img_names[k] for k in exact_match_clean_indicies]

print 'image_model_ids:', image_model_ids
image_ids_for_315_models = []
for modelid in exact_match_modelIds:
  t = []
  for i in range(len(image_model_ids)):
    if modelid == image_model_ids[i]:
      t.append(i)
  image_ids_for_315_models.append(t)

#
# COMPUTE IMAGE-SHAPE DISTANCE MATRIX
#
# configuration params
if args.distance_matrix_txt_file is not None:
  D = np.loadtxt(args.distance_matrix_txt_file)
  if D.shape[0] > len(image_model_ids):
    assert(D.shape[0] == 315)
    if args.clutter_only:
      D = D[exact_match_cluttered_indicies,:] # 141*6777
    elif args.clean_only:
      D = D[exact_match_clean_indicies,:]
else:
  deploy_file = args.model_deploy_file
  shape_embedding_file = args.shape_embedding_file
  embedding_model_path = args.model_param_file
  
  # get image embedding from pool5 features
  feat_name = os.path.join(RESULTDIR, 'tmp_image_embedding')
  cmd = os.path.join(BASEDIR, '/extract_feature_batch_generic.py')
  subprocess.call(['python', cmd, '-i', args.input_npy_file, '-d', deploy_file, '-p', embedding_model_path, '-b', str(50), '--feat_dim', str(args.feat_dim), '--feat_name', 'fc8_embedding', '--save_file', feat_name, '--save_file_format', 'npy'])
  
  # compute distances between images and shapes
  image_embedding = np.load(feat_name+'.npy')
  if args.clutter_only:
    image_embedding = image_embedding[exact_match_cluttered_indicies,:]
  elif args.clean_only:
    image_embedding = image_embedding[exact_match_clean_indicies,:]
  image_embedding = image_embedding.reshape((image_embedding.shape[0], image_embedding.shape[1]))
  assert(image_model_ids.shape[0] == image_embedding.shape[0])
  shape_embedding = np.loadtxt(shape_embedding_file)
  print image_embedding.shape, shape_embedding.shape
  D = cdist(image_embedding, shape_embedding)

#
# IMAGE2SHAPE
#
dist_name = os.path.join(RESULTDIR, 'tmp_image2shape_dist.txt')
np.savetxt(dist_name, D)

print np.shape(D)
image_N = D.shape[0]

image2shape_retrieval_ranking = []
image2shape_retrieval_ranking_105models = []
for k in range(image_N):
  distances = D[k,:]#[float(distance) for distance in line.strip().split()]
  ranking = range(len(distances))
  ranking.sort(key = lambda rank:distances[rank])
  print 'image %d %s \t retrieval: %d' % (k,img_names[k].split('/')[-1], ranking.index(image_model_ids[k])+1)
  image2shape_retrieval_ranking.append(ranking.index(image_model_ids[k])+1)

  # only consider the 105 models
  distances_105models = D[k,exact_match_modelIds]
  ranking_105models = range(len(distances_105models))
  ranking_105models.sort(key = lambda rank:distances_105models[rank])
  image2shape_retrieval_ranking_105models.append(ranking_105models.index(exact_match_modelIds.index(image_model_ids[k]))+1)

 
image2shape_topK_accuracies = []
image2shape_topK_accuracies_105models = []
for topK in range(250):
  n = sum([r <= topK+1 for r in image2shape_retrieval_ranking])
  image2shape_topK_accuracies.append(n / float(image_N))
  
  # only consider the 105 models
  n = sum([r <= topK+1 for r in image2shape_retrieval_ranking_105models])
  image2shape_topK_accuracies_105models.append(n / float(image_N))

np.savetxt(args.result_id+'_image2shape_topK_accuracy.txt', image2shape_topK_accuracies, fmt='%.4f')
np.savetxt(args.result_id+'_image2shape_topK_accuracy_105models.txt', image2shape_topK_accuracies_105models, fmt='%.4f')


#
# SHAPE2IMAGE
#
dist_name = os.path.join(RESULTDIR, 'tmp_shape2image_dist.txt')
np.savetxt(dist_name, D.transpose())

image_model_ids_set = set(image_model_ids)
model_N = min(len(exact_match_modelIds), len(set(image_model_ids)))

first_ranks = []
last_ranks = []

shape2image_retrieval_ranking = []
for k in range(len(exact_match_modelIds)): # 0 - 104
  modelId = exact_match_modelIds[k]

  if modelId not in image_model_ids_set:
    continue

  distances = D.transpose()[modelId,:] # clutter: 141*1
  ranking = range(len(distances))
  ranking.sort(key = lambda rank:distances[rank])
  ranks = [ranking.index(j)+1 for j in image_ids_for_315_models[k]]
  retrieval_rank = min(ranks) # find images corresponding to this model
  print 'model %d %s\t retrieval: %d' % (k,exact_match_modelIds[k], retrieval_rank)
  shape2image_retrieval_ranking.append(retrieval_rank)

  first_ranks.append(min(ranks))
  last_ranks.append(max(ranks))

shape2image_topK_accuracies = []
for topK in range(250):
  n = sum([r <= topK+1 for r in shape2image_retrieval_ranking])
  shape2image_topK_accuracies.append(n / float(model_N))

print first_ranks
print last_ranks
np.savetxt(args.result_id+'_shape2image_topK_accuracy.txt', shape2image_topK_accuracies, fmt='%.4f')
np.savetxt(args.result_id+'_first_last_appearance_median_rank.txt', [np.median(first_ranks), np.median(last_ranks)], fmt='%d')
