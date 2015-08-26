#!/bin/bash

first=
last=

usage() { echo "run preparation pipeline from first to last step specified by -f and -l option."; }

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
  echo "-l option is not presented, run the pipeline until last=3!"
  last=3;
fi



# Step 00
# Edit global_variables.py

# Step 01
# Prepare 3rd party executables/libraries
if [ "$first" -le 1 ] && [ "$last" -ge 1 ]; then
  python ./prepare_3rd_party.py
fi

# Step 02
# Prepare shell scripts
if [ "$first" -le 2 ] && [ "$last" -ge 2 ]; then
  python ./prepare_shell_scripts.py
fi

# Step 03
# Generate shape list
if [ "$first" -le 3 ] && [ "$last" -ge 3 ]; then
  python ./generate_shape_list.py;
fi
