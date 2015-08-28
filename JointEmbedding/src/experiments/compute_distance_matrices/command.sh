#!/bin/bash

BASEDIR="$(dirname $(readlink -f $0))"
EXACTMATCH_DATASET=$BASEDIR/../ExactMatchChairsDataset


# extract hog features and compute distance matrix
matlab -nodisplay -r get_hog_feat_distance_matrix

# extract pure image pool5
python $BASEDIR/extract_feature_batch.py -i $BASEDIR/sampled_pure_img_filelist.txt -d $BASEDIR/deploy.prototxt -p $BASEDIR/bvlc_reference_rcnn_ilsvrc13.caffemodel -b 500 --feat_dim 9216 --feat_name pool5 --save_file $BASEDIR/pure_img_pool5_feat --resize_dim 227 --save_file_format npy --gpu_index 0

# extract pure image siamese embedding and compute distance matrix
python $BASEDIR/../extract_feature_batch_generic.py -i $BASEDIR/pure_img_pool5_feat.npy -d $BASEDIR/extract_siamese_image_embedding_clean.prototxt -p $BASEDIR/siamese_contrastive_loss.caffemodel  -b 500 --feat_dim 512 --feat_name image_embedding --save_file $BASEDIR/pure_img_siamese_embedding --gpu_index 0

python $BASEDIR/../extract_feature_batch_generic.py -i $EXACTMATCH_DATASET/exact_match_chairs_pool5_feat.npy -d $BASEDIR/extract_siamese_image_embedding_clean.prototxt -p $BASEDIR/siamese_contrastive_loss.caffemodel  -b 500 --feat_dim 512 --feat_name image_embedding --save_file $BASEDIR/exact_match_chairs_siamese_embedding --gpu_index 0

python $BASEDIR/get_siamese_embedding_distance_matrix.py

# extract alexnet fc7 feature and compute distance matrix

python $BASEDIR/extract_feature_batch.py -i $EXACTMATCH_DATASET/exact_match_chairs_img_filelist.txt -d $BASEDIR/deploy.prototxt -p $BASEDIR/bvlc_reference_rcnn_ilsvrc13.caffemodel -b 500 --feat_dim 4096 --feat_name fc7 --save_file $BASEDIR/exact_match_fc7_feat --resize_dim 227 --save_file_format npy --gpu_index 0

python $BASEDIR/extract_feature_batch.py -i $BASEDIR/pure_img_filelist.txt -d $BASEDIR/deploy.prototxt -p $BASEDIR/bvlc_reference_rcnn_ilsvrc13.caffemodel -b 500 --feat_dim 4096 --feat_name fc7 --save_file pure_img_fc7_feat --resize_dim 227 --save_file_format npy --gpu_index 0

python $BASEDIR/get_cnn_fc7_feat_distance_matrix.py
