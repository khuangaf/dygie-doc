#!/usr/bin/env bash
'''
Copied over from https://github.com/fenchri/edge-oriented-graph.
'''
pwd=$PWD
out_dir=data/cdr
script_dir=scripts/data/cdr
raw_dir=$out_dir/raw-data/CDR_Data
intermediate_dir=$out_dir/intermediate
processed_dir=$out_dir/processed-data
common_dir=scripts/data/common
mkdir $intermediate_dir
mkdir $processed_dir
mkdir $common_dir


# cd $common_dir
# wget http://www.nactem.ac.uk/y-matsu/geniass/geniass-1.00.tar.gz && tar xvzf geniass-1.00.tar.gz
# cd geniass/ && make && cd ..

# # this forks contain the fixed Makefile
# git clone https://github.com/khuangaf/genia-tagger-py
# cd genia-tagger-py 
# make
# cd $pwd


# # set path for necessary GENIA TAGGER library
# export PYTHONPATH=$PWD/scripts/data/common/genia-tagger-py
# export PATH=$PATH:$PWD/scripts/data/common/geniass/
# for d in "Training" "Development" "Test";
# do
#     mkdir $intermediate_dir/${d}
#     python $script_dir/process.py --input_file $raw_dir/CDR.Corpus.v010516/CDR_${d}Set.PubTator.txt \
#                        --output_file $intermediate_dir/${d} \
#                        --data CDR

#     python $script_dir/filter_hypernyms.py --mesh_file $script_dir/2017MeshTree.txt \
#                                 --input_file $intermediate_dir/${d}.data \
#                                 --output_file $intermediate_dir/${d}_filter.data
    
# done

python $script_dir/convert_cdr2dygie.py --input_path $intermediate_dir/Training_filter.data --output_path $processed_dir/train.json --split train
python $script_dir/convert_cdr2dygie.py --input_path $intermediate_dir/Development_filter.data --output_path $processed_dir/dev.json --split dev
python $script_dir/convert_cdr2dygie.py --input_path $intermediate_dir/Test_filter.data --output_path $processed_dir/test.json --split test