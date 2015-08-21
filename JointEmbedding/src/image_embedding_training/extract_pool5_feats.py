import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

script_path = os.path.join(BASE_DIR, 'image_embedding_training/extract_feature_batch.py')
cmd = 'python %s --img_filelist %s --deploy_file %s --params_file %s --feat_name pool5 --lmdb %s --gpu_index %d --caffe_path %s --pool_size %d' % \
(script_path, g_syn_images_filelist, g_extract_feat_deploy_file, g_fine_tune_caffemodel_file, g_pool5_lmdb, g_extract_feat_gpu_index, g_caffe_executable_path, g_extract_feat_thread_num)

os.system(cmd)
