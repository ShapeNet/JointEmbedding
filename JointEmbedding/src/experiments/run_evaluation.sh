#!/bin/bash

shape_embedding_file='/orions3-zfs/projects/haosu/Image2Scene/data/chair6777_meta/modelFeaturesSammon100.txt'
model_deploy_file='/orions3-zfs/projects/haosu/Image2Scene/data/model_chair_view_prediction_7k_siamese/fine_tune_sammon100/deploy.prototxt'
model_param_file='/orions3-zfs/projects/haosu/Image2Scene/data/model_chair_view_prediction_7k_siamese/fine_tune_sammon100/snapshots_iter_10000.caffemodel'

BASEDIR="$(dirname $(readlink -f $0))"
RESULTDIR=$BASEDIR/results
EXACTMATCH_DATASET=$BASEDIR/ExactMatchChairsDataset
DISTANCE_MATRIX_DIR=$BASEDIR/compute_distance_matrices
if ! [ -e $RESULTDIR ]; then
    mkdir $RESULTDIR
fi

# figure 10 our embedding
python $BASEDIR/shape_retrieval_by_images_evaluation.py -m $EXACTMATCH_DATASET/exact_match_chairs_img_modelIds_0to6776.txt -i $EXACTMATCH_DATASET/exact_match_chairs_pool5_feat.npy -s $shape_embedding_file -d $model_deploy_file -p $model_param_file -n1 250 -n2 250 --result_id $RESULTDIR/sammon100_clutter --clutter_only
# table 2 our embedding
python $BASEDIR/shape_retrieval_by_images_evaluation.py -m $EXACTMATCH_DATASET/exact_match_chairs_img_modelIds_0to6776.txt -i $EXACTMATCH_DATASET/exact_match_chairs_pool5_feat.npy -s $shape_embedding_file -d $model_deploy_file -p $model_param_file -n1 250 -n2 250 --result_id $RESULTDIR/sammon100_tmp


