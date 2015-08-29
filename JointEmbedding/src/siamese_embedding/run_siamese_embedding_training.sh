#!/bin/bash

first=
last=

usage() { echo "run siamese embedding training pipeline from first to last step specified by -f and -l option."; }

first_flag=0
last_flag=0
while getopts f:l:h opt; do
  case $opt in
  f)
    first_flag=1;
    first=$(($OPTARG))
    ;;
  l)
    last_flag=1;
    last=$(($OPTARG))
    ;;
  h)
    usage; exit;;
  esac
done

shift $((OPTIND-1))

if [ $first_flag -eq 0 ]
then
  echo "-f option is not presented, run the pipeline from first=1!"
  first=1;
fi

if [ $last_flag -eq 0 ]
then
  echo "-l option is not presented, run the pipeline until last=4!"
  last=4;
fi


# Step 01
# Sample image pages and gnerate caffe training scripts
if [ "$first" -le 1 ] && [ "$last" -ge 1 ]; then
  python ./prepare_training.py
fi

# Step 02
# Split train/val
if [ "$first" -le 2 ] && [ "$last" -ge 2 ]; then
  python ./split_train_val.py
fi

# Step 03
# Generate train/val lmdbs of synthetic images
if [ "$first" -le 3 ] && [ "$last" -ge 3 ]; then
  python ./gen_syn_images_pairs_lmdbs.py
fi
