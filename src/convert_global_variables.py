#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from global_variables import *

global_variables_m_file = open('./global_variables.m', 'w')
global_variables_m_file.write('g_shapenet_synset_set = {%s};\n' %(''.join(['\''+synset+'\'' for synset in g_shapenet_synset_set])))
global_variables_m_file.write('\n');
global_variables_m_file.write('g_piotr_toolbox_path = \'%s\';\n' %(g_piotr_toolbox_path))
global_variables_m_file.write('g_minfunc_2012_path = \'%s\';\n' %(g_minfunc_2012_path))
global_variables_m_file.write('g_shape_list_file = \'%s\';\n' %(g_shape_list_file))
global_variables_m_file.write('\n');
global_variables_m_file.write('g_lfd_cropping_thread_num = %d;\n' %(g_lfd_cropping_thread_num))
global_variables_m_file.write('g_lfd_images_folder = \'%s\';\n' %(g_lfd_images_folder))
global_variables_m_file.write('g_lfd_images_cropped_folder = \'%s\';\n' %(g_lfd_images_cropped_folder))
global_variables_m_file.write('\n');
global_variables_m_file.write('g_lfd_view_num = %d;\n' %(g_lfd_view_num))
global_variables_m_file.write('g_lfd_hog_extraction_thread_num = %d;\n' %(g_lfd_hog_extraction_thread_num))
global_variables_m_file.write('g_lfd_hog_image_size = %d;\n' %(g_lfd_hog_image_size))
global_variables_m_file.write('g_lfd_hog_features_file = \'%s\';\n' %(g_lfd_hog_features_file))
global_variables_m_file.write('g_shape_distance_matrix_file_mat = \'%s\';\n' %(g_shape_distance_matrix_file_mat))
global_variables_m_file.write('g_shape_distance_matrix_file_txt = \'%s\';\n' %(g_shape_distance_matrix_file_txt))
global_variables_m_file.write('\n');
global_variables_m_file.write('g_shape_embedding_space_dimension = %d;\n' %(g_shape_embedding_space_dimension))
global_variables_m_file.write('g_shape_embedding_space_file_mat = \'%s\';\n' %(g_shape_embedding_space_file_mat))
global_variables_m_file.write('g_shape_embedding_space_file_txt = \'%s\';\n' %(g_shape_embedding_space_file_txt))
global_variables_m_file.write('\n');
global_variables_m_file.write('g_syn_cropping_thread_num = %d;\n' %(g_syn_cropping_thread_num))
global_variables_m_file.write('g_syn_images_folder = \'%s\';\n' %(g_syn_images_folder))
global_variables_m_file.write('g_syn_images_cropped_folder = \'%s\';\n' %(g_syn_images_cropped_folder))
global_variables_m_file.write('g_syn_bkg_overlay_thread_num = %d;\n' %(g_syn_bkg_overlay_thread_num))
global_variables_m_file.write('g_syn_images_bkg_overlaid_folder = \'%s\';\n' %(g_syn_images_bkg_overlaid_folder))
global_variables_m_file.write('g_syn_bkg_filelist = \'%s\';\n' %(g_syn_bkg_filelist))
global_variables_m_file.write('g_syn_bkg_folder = \'%s\';\n' %(g_syn_bkg_folder))
global_variables_m_file.write('g_syn_cluttered_bkg_ratio = %f;\n' %(g_syn_cluttered_bkg_ratio))

global_variables_m_file.write('g_data_folder = \'%s\';\n' %(g_data_folder))
global_variables_m_file.write('g_shapenet_synset_set_handle = \'%s\';\n' %(g_shapenet_synset_set_handle))
global_variables_m_file.write('g_mirror_name = \'%s\';\n' %(g_mirror_name))
global_variables_m_file.close();
