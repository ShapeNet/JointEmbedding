#!/bin/bash

shape_embedding_file='/orions3-zfs/projects/haosu/Image2Scene/data/chair6777_meta/modelFeaturesSammon100.txt'
model_deploy_file='/orions3-zfs/projects/haosu/Image2Scene/data/model_chair_view_prediction_7k_siamese/fine_tune_sammon100/deploy.prototxt'
model_param_file='/orions3-zfs/projects/haosu/Image2Scene/data/model_chair_view_prediction_7k_siamese/fine_tune_sammon100/snapshots_iter_10000.caffemodel'

BASEDIR="$(dirname $(readlink -f $0))"
mkdir $BASEDIR/results

# figure 10 our embedding
python ./shape_retrieval_by_images_evaluation.py -m /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_img_modelIds_0to6776.txt -i /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_pool5_feat.npy -s $shape_embedding_file -d $model_deploy_file -p $model_param_file -n1 250 -n2 250 --result_id sammon100_clutter --clutter_only
exit
# table 2 our embedding
python ./shape_retrieval_by_images_evaluation.py -m /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_img_modelIds_0to6776.txt -i /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_pool5_feat.npy -s $shape_embedding_file -d $model_deploy_file -p $model_param_file -n1 250 -n2 250 --result_id sammon100_tmp


hog_withoutview_distance_matrix_txt_file='/orions3-zfs/projects/haosu/Image2Scene/code/matlab/evaluation/hogImageModelDist_withoutViewOracle.txt'
# figure 10 hog
python ./shape_retrieval_by_images_evaluation.py -m /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_img_modelIds_0to6776.txt -i /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_pool5_feat.npy -n1 250 -n2 250 --distance_matrix_txt_file $hog_withoutview_distance_matrix_txt_file --result_id hog_withoutview_clutter --clutter_only
# table 2 hog
python ./shape_retrieval_by_images_evaluation.py -m /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_img_modelIds_0to6776.txt -i /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_pool5_feat.npy -n1 250 -n2 250 --distance_matrix_txt_file $hog_withoutview_distance_matrix_txt_file --result_id hog_withoutview_tmp


cnn_pool5_withoutview_distance_matrix_txt_file='cnn_feature/CNN_pool5_distMatrix_withoutViewOracle.txt'
# figure 10 pool5
python ./shape_retrieval_by_images_evaluation.py -m /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_img_modelIds_0to6776.txt -i /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_pool5_feat.npy -n1 250 -n2 250 --distance_matrix_txt_file $cnn_pool5_withoutview_distance_matrix_txt_file --result_id cnn_pool5_withoutview_clutter --clutter_only
# table 2 pool5
python ./shape_retrieval_by_images_evaluation.py -m /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_img_modelIds_0to6776.txt -i /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_pool5_feat.npy -n1 250 -n2 250 --distance_matrix_txt_file $cnn_pool5_withoutview_distance_matrix_txt_file --result_id cnn_pool5_withoutview_tmp

cnn_fc7_withoutview_distance_matrix_txt_file='cnn_feature/CNN_fc7_distMatrix_withoutViewOracle.txt'
# figure 10 fc7
python ./shape_retrieval_by_images_evaluation.py -m /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_img_modelIds_0to6776.txt -i /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_pool5_feat.npy -n1 250 -n2 250 --distance_matrix_txt_file $cnn_fc7_withoutview_distance_matrix_txt_file --result_id cnn_fc7_withoutview_clutter --clutter_only
# table 2 fc7
python ./shape_retrieval_by_images_evaluation.py -m /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_img_modelIds_0to6776.txt -i /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_pool5_feat.npy -n1 250 -n2 250 --distance_matrix_txt_file $cnn_fc7_withoutview_distance_matrix_txt_file --result_id cnn_fc7_withoutview_tmp


siamese_20view_distance_matrix_txt_file='siamese/siamese_distMatrix_withoutViewOracle.txt'
# figure 10 siamese
python /orions3-zfs/projects/haosu/Image2Scene/code/python/shape_retrieval_by_images_evaluation.py -m /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_img_modelIds_0to6776.txt -i /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_pool5_feat.npy -n1 250 -n2 250 --distance_matrix_txt_file $siamese_20view_distance_matrix_txt_file --result_id siamese_20view_clutter --clutter_only
# table 2 siamese
python /orions3-zfs/projects/haosu/Image2Scene/code/python/shape_retrieval_by_images_evaluation.py -m /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_img_modelIds_0to6776.txt -i /orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_pool5_feat.npy -n1 250 -n2 250 --distance_matrix_txt_file $siamese_20view_distance_matrix_txt_file --result_id siamese_20view_tmp
