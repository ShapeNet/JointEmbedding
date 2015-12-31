#!/usr/bin/python
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Define all the global variables for the project
# Many of the variables are just for organizing the files/folders in a nice
# way. Variables that you should take care of are marked by "[take care!]"
# tags. Good luck!
#------------------------------------------------------------------------------

import os
SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
g_data_folder = os.path.abspath(os.path.join(SRC_ROOT, '../data'))

##############################################################################
# Paths and urls to executable and data sources
##############################################################################
g_3rd_party_folder = os.path.abspath(os.path.join(SRC_ROOT, '3rd_party'))
g_blender_executable_url = 'http://download.blender.org/release/Blender2.75/blender-2.75a-linux-glibc211-x86_64.tar.bz2'
g_blender_executable_path = os.path.abspath(os.path.join(g_3rd_party_folder, 'blender/blender'))
g_piotr_toolbox_git = 'https://github.com/pdollar/toolbox'
g_piotr_toolbox_path = os.path.abspath(os.path.join(g_3rd_party_folder, 'piotr_toolbox'))
g_minfunc_2012_url = 'https://www.cs.ubc.ca/~schmidtm/Software/minFunc_2012.zip'
g_minfunc_2012_path = os.path.abspath(os.path.join(g_3rd_party_folder, 'minFunc_2012'))
# Follow Caffe homepage for the installation instructions of Caffe.
g_caffe_install_path = os.path.abspath('/opt/caffe') # [take care!!!]
g_matlab_executable_path = os.path.abspath('/usr/local/bin/matlab') # [take care!!!]
g_shapenet_root_folder = os.path.join(g_data_folder, 'ShapeNetCore2015Summer') # [take care!!!], where you put ShapeNet data
g_sun2012_data_url = 'http://groups.csail.mit.edu/vision/SUN/releases/SUN2012pascalformat.tar.gz'

# We fine tune our model from RCNN model
g_network_architecture_name = 'rcnn'
caffemodel_url_handle_dict = dict()
caffemodel_url_handle_dict['alexnet'] = 'bvlc_alexnet'
caffemodel_url_handle_dict['rcnn'] = 'bvlc_reference_rcnn_ilsvrc13'
g_fine_tune_caffemodel_url = 'http://dl.caffe.berkeleyvision.org/'+caffemodel_url_handle_dict[g_network_architecture_name]+'.caffemodel'
g_fine_tune_caffemodel = os.path.join(g_data_folder, 'image_embedding/'+caffemodel_url_handle_dict[g_network_architecture_name]+'.caffemodel')
g_mean_file = os.path.join(SRC_ROOT, 'ilsvrc_2012_mean.npy')

g_thread_num = 32 # [take care!!!], try to match with #CPU core


##############################################################################
# Specify which categories will be used for generating the embedding space
##############################################################################
all_shapenet_synset_set = ['03001627' # chair
                          ,'04379243' # table
                          ,'03636649' # lamp
                          ,'02958343' # car
                          ,'02691156' # airplane
                           ]

#-----------------------------------------------------------------------------
# Multiple categories can be put into a single embedding space. However, it
# will result in a huge distance matrix, both the distance matrix computetion
# and MDS embedding for it will takes very long time to compute.
#-----------------------------------------------------------------------------
#g_shapenet_synset_set = ['03001627', '02958343', '02691156'] # chair, car, airplane

#-----------------------------------------------------------------------------
# In practice, just train one model for each category...
# Since there are DL models that do great job on object classification.
# The practical pipeline can be:
# Input image --> classifier --> class specific joint embedding
#-----------------------------------------------------------------------------
g_shapenet_synset_set = ['03001627'] # chair
#g_shapenet_synset_set = ['04379243'] # table
#g_shapenet_synset_set = ['03636649'] # lamp
#g_shapenet_synset_set = ['02958343'] # car
#g_shapenet_synset_set = ['02691156'] # airplane

# Suffix generated intermediate result with the synset_set
g_shapenet_synset_set_handle = '_'+'_'.join(g_shapenet_synset_set)

# This is for creating a "mirror" dataset for various experiments.
# For example, exclude some shapes in the training...
g_mirror_mode = False
g_mirror_name = 'no_exact_match'
if g_mirror_mode:
    g_shapenet_synset_set_handle = g_shapenet_synset_set_handle + '_' + g_mirror_name;
    
g_shape_list_file = os.path.join(g_data_folder, 'shape_list'+g_shapenet_synset_set_handle+'.txt')


##############################################################################
# Shape embedding
##############################################################################
# Rendering
g_lfd_light_num = 4
g_lfd_light_dist = 14.14
g_lfd_camera_dist = 3
g_lfd_view_num = 20 #[take care!] g_lfd_view_num = elevation_num*azimuth_num
g_lfd_rendering_thread_num = g_thread_num #[take care!], try to match with #CPU core
#-----------------------------------------------------------------------------
# The elevations and azimuths are category specific
# We simply manually pick k (k=20) LFD views for each category, A potentially
# better approach could be render M (M>>k) views, and automatically select
# the most informative k views out of the M views.
#-----------------------------------------------------------------------------
# [take care!], if you cadd new categories.
g_lfd_camera_elevation_dict = {
'03001627': [20],
'02958343': [10],
'02691156': [-30, 0],
'04379243': [20],
'03636649': [-30, 0]
}
# [take care!], if you cadd new categories.
g_lfd_camera_azimuth_dict = {
'03001627': [x*18 for x in range(0,20)],
'02958343': [x*18 for x in range(0,20)],
'02691156': [x*36 for x in range(0,10)],
'04379243': [x*18 for x in range(0,20)],
'03636649': [x*36 for x in range(0,10)]
}
g_lfd_images_folder = os.path.join(g_data_folder, 'shape_embedding/lfd_images')

# Cropping
# Consider change the default parfor worker number in matlab by following the instructions here:
# http://www.mathworks.com/help/distcomp/saveprofile.html
g_lfd_cropping_thread_num = g_thread_num # [take care!], try to match with #CPU core
g_lfd_images_cropped_folder = os.path.join(g_data_folder, 'shape_embedding/lfd_images_cropped')

# HoG feature extraction
# Consider change the default parfor worker number in matlab by following the instructions here:
# http://www.mathworks.com/help/distcomp/saveprofile.html
g_lfd_hog_extraction_thread_num = g_thread_num # [take care!], try to match with #CPU core
g_lfd_hog_image_size = 120
g_lfd_hog_features_file = os.path.join(g_data_folder, 'shape_embedding/lfd_hog_features'+g_shapenet_synset_set_handle+'.mat')

# Pairwise distance
g_shape_distance_matrix_file_mat = os.path.join(g_data_folder, 'shape_embedding/shape_distance_matrix'+g_shapenet_synset_set_handle+'.mat')
g_shape_distance_matrix_file_txt = os.path.join(g_data_folder, 'shape_embedding/shape_distance_matrix'+g_shapenet_synset_set_handle+'.txt')

# Embedding space
g_shape_embedding_space_dimension = 128
g_shape_embedding_space_file_mat = os.path.join(g_data_folder, 'shape_embedding/shape_embedding_space'+g_shapenet_synset_set_handle+'.mat')
g_shape_embedding_space_file_txt = os.path.join(g_data_folder, 'shape_embedding/shape_embedding_space'+g_shapenet_synset_set_handle+'.txt')


##############################################################################
# Image embedding
##############################################################################
# Rendering
g_blank_blend_file_path = os.path.join(SRC_ROOT, 'blank.blend')
g_syn_rendering_thread_num = g_thread_num #[take care!], try to match with #CPU core
g_syn_images_per_shape = 125
g_view_distribution_folder = os.path.join(g_data_folder, 'image_embedding/view_distribution')
g_view_distribution_files = dict(zip(all_shapenet_synset_set, [os.path.join(g_view_distribution_folder, synset+'.txt') for synset in all_shapenet_synset_set]))
g_view_distribution_params = dict()
#-----------------------------------------------------------------------------
# The camera view distributions are category specific
# Ideally, they should be learned from large image collections with camera view annotations.
# Otherwise, just try to specify a rough approximation.
# First list is for azimuth, 16 slots each for [22.5*i-11.25,22.5*i+11.25]
# Second list is for elevation, 18 slots each for [10*i-90,10*i-80]
# Last number is for tilt deviation, normal(0,s)
# Note that, the scale of the values in the two list doesn't matter
#-----------------------------------------------------------------------------
g_view_distribution_params['03001627'] = [[93, 90, 79, 66, 56, 49, 45, 44, 45, 46, 48, 51, 56, 63, 75, 87],[1, 1, 1, 1, 1, 1, 1, 23, 57, 212, 315, 181, 80, 36, 26, 21, 18, 20],3]
g_view_distribution_params['02958343'] = [[91, 87, 73, 58, 50, 50, 56, 65, 69, 62, 52, 44, 42, 49, 64, 81],[0, 0, 0, 0, 0, 0, 0, 24, 227, 521, 86, 26, 18, 15, 18, 10, 10, 10],3]
g_view_distribution_params['02691156'] = [[65, 70, 74, 75, 71, 64, 55, 48, 45, 47, 53, 61, 66, 67, 65, 64],[19, 13, 15, 18, 23, 35, 64, 125, 223, 216, 95, 33, 22, 22, 18, 15, 15, 19],10]
g_view_distribution_params['04379243'] = [[10]*16,[1, 1, 1, 1, 1, 1, 1, 23, 57, 212, 315, 181, 80, 36, 26, 21, 18, 20],3]
g_view_distribution_params['03636649'] = [[10]*16,[10]*18,10]
g_syn_light_num_lowbound = 0
g_syn_light_num_highbound = 4
g_syn_light_dist_lowbound = 8
g_syn_light_dist_highbound = 20
g_syn_camera_dist = 3
g_syn_images_folder = os.path.join(g_data_folder, 'image_embedding/syn_images')

# Cropping
# Consider change the default parfor worker number in matlab by following the instructions here:
# http://www.mathworks.com/help/distcomp/saveprofile.html
g_syn_cropping_thread_num = g_thread_num # [take care!], try to match with #CPU core
g_syn_images_cropped_folder = os.path.join(g_data_folder, 'image_embedding/syn_images_cropped')

# Background overlay
# Consider change the default parfor worker number in matlab by following the instructions here:
# http://www.mathworks.com/help/distcomp/saveprofile.html
g_syn_bkg_overlay_thread_num = g_thread_num #[take care!], try to match with #CPU core
g_syn_cluttered_bkg_ratio = 0.9
g_sun2012_data_folder = os.path.join(g_data_folder, 'image_embedding/sun2012_data')
g_syn_bkg_filelist = os.path.join(g_sun2012_data_folder, 'filelist.txt');
g_syn_bkg_folder = os.path.join(g_sun2012_data_folder, 'JPEGImages');
g_syn_images_bkg_overlaid_folder = os.path.join(g_data_folder, 'image_embedding/syn_images_bkg_overlaid')

# Filelists, train/val splits
g_train_ratio = 0.9
g_syn_images_filelist = os.path.join(g_data_folder, 'image_embedding/syn_images_filelist'+g_shapenet_synset_set_handle+'.txt')
g_syn_images_filelist_train = os.path.join(g_data_folder, 'image_embedding/syn_images_filelist'+g_shapenet_synset_set_handle+'_train.txt')
g_syn_images_filelist_val = os.path.join(g_data_folder, 'image_embedding/syn_images_filelist'+g_shapenet_synset_set_handle+'_val.txt')
g_syn_images_imageid2shapeid = os.path.join(g_data_folder, 'image_embedding/syn_images_imageid2shapeid'+g_shapenet_synset_set_handle+'.txt')
g_syn_images_imageid2shapeid_train = os.path.join(g_data_folder, 'image_embedding/syn_images_imageid2shapeid'+g_shapenet_synset_set_handle+'_train.txt')
g_syn_images_imageid2shapeid_val = os.path.join(g_data_folder, 'image_embedding/syn_images_imageid2shapeid'+g_shapenet_synset_set_handle+'_val.txt')
g_syn_images_train_val_split = os.path.join(g_data_folder, 'image_embedding/syn_images_train_val_split'+g_shapenet_synset_set_handle+'.txt')

# Pool5 features
g_pool5_lmdb = os.path.join(g_data_folder, 'image_embedding/syn_images_pool5_lmdb'+g_shapenet_synset_set_handle+'_'+g_network_architecture_name)
g_pool5_lmdb_train = os.path.join(g_data_folder, 'image_embedding/syn_images_pool5_lmdb'+g_shapenet_synset_set_handle+'_'+g_network_architecture_name+'_train')
g_pool5_lmdb_val = os.path.join(g_data_folder, 'image_embedding/syn_images_pool5_lmdb'+g_shapenet_synset_set_handle+'_'+g_network_architecture_name+'_val')
g_extract_feat_prototxt = os.path.join(SRC_ROOT, 'image_embedding_training/pool5_feature_extraction_'+g_network_architecture_name+'.prototxt')
# Consider change the default parfor worker number in matlab by following the instructions here:
# http://www.mathworks.com/help/distcomp/saveprofile.html
g_extract_feat_thread_num = g_thread_num #[take care!], try to match with #CPU core
g_extract_feat_gpu_index = 0 #[take care!], which GPU to use for pool5 extraction

# Shape embedding LMDBs
g_shape_embedding_lmdb_train = os.path.join(g_data_folder, 'shape_embedding/shape_embedding_lmdb'+g_shapenet_synset_set_handle+'_train')
g_shape_embedding_lmdb_val = os.path.join(g_data_folder, 'shape_embedding/shape_embedding_lmdb'+g_shapenet_synset_set_handle+'_val')

# Train caffemodel
g_image_embedding_training_folder = os.path.join(g_data_folder, 'image_embedding/image_embedding_training'+g_shapenet_synset_set_handle+'_'+g_network_architecture_name)
if not os.path.exists(g_image_embedding_training_folder):
    os.makedirs(g_image_embedding_training_folder)
g_image_embedding_train_val_prototxt = os.path.join(g_image_embedding_training_folder, 'train_val_'+g_network_architecture_name+'.prototxt') 
g_image_embedding_solver_prototxt = os.path.join(g_image_embedding_training_folder, 'solver.prototxt')
g_image_embedding_command_sh = os.path.join(g_image_embedding_training_folder, 'run_training.sh')

# Testing
g_image_embedding_testing_folder = os.path.join(g_data_folder, 'image_embedding/image_embedding_testing'+g_shapenet_synset_set_handle+'_'+g_network_architecture_name)
if not os.path.exists(g_image_embedding_testing_folder):
    os.makedirs(g_image_embedding_testing_folder)
g_image_embedding_testing_prototxt = os.path.join(g_image_embedding_testing_folder, 'image_embedding_'+g_network_architecture_name+'.prototxt')


##############################################################################
# Siamese embedding (for comparison to our approach)
##############################################################################
g_siamese_pair_top_k = 64
top_k_handle = '_top_'+str(g_siamese_pair_top_k)
g_siamese_pairs_num = 8000000
g_syn_images_pairs_filelist = os.path.join(g_data_folder, 'siamese_embedding/syn_images_pairs_filelist'+g_shapenet_synset_set_handle+top_k_handle+'.txt')
g_syn_images_pairs_train_val_split = os.path.join(g_data_folder, 'siamese_embedding/syn_images_pairs_train_val_split'+g_shapenet_synset_set_handle+top_k_handle+'.txt')
g_syn_images_pairs_filelist_train = os.path.join(g_data_folder, 'siamese_embedding/syn_images_pairs_filelist'+g_shapenet_synset_set_handle+top_k_handle+'_train.txt')
g_syn_images_pairs_filelist_val = os.path.join(g_data_folder, 'siamese_embedding/syn_images_pairs_filelist'+g_shapenet_synset_set_handle+top_k_handle+'_val.txt')

g_gen_siamese_lmdb_thread_num = g_thread_num #[take care!], try to match with #CPU core
g_pairs_pool5_lmdb_train = os.path.join(g_data_folder, 'siamese_embedding/syn_images_pairs_pool5_lmdb'+g_shapenet_synset_set_handle+top_k_handle+'_'+g_network_architecture_name+'_train')
g_pairs_pool5_lmdb_val = os.path.join(g_data_folder, 'siamese_embedding/syn_images_pairs_pool5_lmdb'+g_shapenet_synset_set_handle+top_k_handle+'_'+g_network_architecture_name+'_val')

g_siamese_embedding_training_folder = os.path.join(g_data_folder, 'siamese_embedding/siamese_embedding_training'+g_shapenet_synset_set_handle+top_k_handle+'_'+g_network_architecture_name)
if not os.path.exists(g_siamese_embedding_training_folder):
    os.makedirs(g_siamese_embedding_training_folder)
g_siamese_embedding_train_val_prototxt = os.path.join(g_siamese_embedding_training_folder, 'train_val_'+g_network_architecture_name+'.prototxt') 
g_siamese_embedding_solver_prototxt = os.path.join(g_siamese_embedding_training_folder, 'solver.prototxt')
g_siamese_embedding_command_sh = os.path.join(g_siamese_embedding_training_folder, 'run_training.sh')
g_siamese_embedding_testing_folder = os.path.join(g_data_folder, 'siamese_embedding/siamese_embedding_testing'+g_shapenet_synset_set_handle+top_k_handle+'_'+g_network_architecture_name)
if not os.path.exists(g_siamese_embedding_testing_folder):
    os.makedirs(g_siamese_embedding_testing_folder)
g_siamese_embedding_testing_prototxt = os.path.join(g_siamese_embedding_testing_folder, 'siamese_embedding_'+g_network_architecture_name+'.prototxt')


##############################################################################
# Experiments (for evaluation)
##############################################################################
# parmas for rendering pure (fixedview) images
g_fixedview_image_per_model = 100
g_fixedview_image_folder = os.path.join(g_data_folder, 'image_embedding/pure_fixedview_images')
g_fixedview_elevation_degs = [0,10,20,30,40]
g_fixedview_elevation_sample_deg = 20
g_fixedview_azimuth_degs = range(0,360,18)
g_fixedview_light_num = 4

