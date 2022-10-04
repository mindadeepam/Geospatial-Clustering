#!/bin/bash

# action=$1
# path=$2

# pip install -r requirements.txt

action=''
path=''
max_size=0
n_clusters=0
# unset n_clusters max_size

print_usage() {
  printf "Usage: ..."
}

while getopts 'a:p:n:m:' flag; do
  case "${flag}" in
    a) action="${OPTARG}";;
    p) path="${OPTARG}" ;;
    n) n_clusters=${OPTARG} ;;
    m) max_size=${OPTARG} ;;
    *) print_usage
        ;;
  esac
done
# echo "action ${action} "
# echo ${action}
# echo "path ${path} "
# echo "max_size ${max_size}"
# echo "n_clusters ${n_clusters}"

python main.py \
  --action=${action} \
  --path_to_data=${path} \
  --n_clusters=${n_clusters} \
  --max_size=${max_size}


#################

# bash ./run.sh train /Users/deepamminda/Documents/Geospatial-Clustering/sample_train_data.csv     
# '''
# POSITIONAL_ARGS=()

# while [[ $# -gt 0 ]]; do
#   case $1 in
#     -e|--extension)
#       EXTENSION="$2"
#       shift # past argument
#       shift # past value
#       ;;
#     -s|--searchpath)
#       SEARCHPATH="$2"
#       shift # past argument
#       shift # past value
#       ;;
#     --default)
#       DEFAULT=YES
#       shift # past argument
#       ;;
#     -*|--*)
#       echo "Unknown option $1"
#       exit 1
#       ;;
#     *)
#       POSITIONAL_ARGS+=("$1") # save positional arg
#       shift # past argument
#       ;;
#   esac
# done

# set -- "${POSITIONAL_ARGS[@]}"


# while [[ "$#" -gt 0 ]]; do
#     case $1 in
#         -t|--target) target="$2"; shift ;;
#         -u|--uglify) uglify=1 ;;
#         *) echo "Unknown parameter passed: $1"; exit 1 ;;
#     esac
#     shift
# done

# echo "Where to deploy: $target"
# echo "Should uglify  : $uglify"  
