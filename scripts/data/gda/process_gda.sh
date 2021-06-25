#!/usr/bin/env bash
'''
Copied over from https://github.com/fenchri/edge-oriented-graph.
'''
pwd=$PWD
out_dir=data/gda
script_dir=scripts/data/gda
raw_dir=$out_dir/raw-data
intermediate_dir=$out_dir/intermediate
processed_dir=$out_dir/processed-data
common_dir=scripts/data/common
mkdir $intermediate_dir
mkdir $processed_dir
mkdir $common_dir


cd $common_dir
wget http://www.nactem.ac.uk/y-matsu/geniass/geniass-1.00.tar.gz && tar xvzf geniass-1.00.tar.gz
rm geniass-1.00.tar.gz
cd geniass/ && make && cd ..

# this forks contain the fixed Makefile
git clone https://github.com/khuangaf/genia-tagger-py
cd genia-tagger-py 
make
cd $pwd


# set path for necessary GENIA TAGGER library
export PYTHONPATH=$PWD/scripts/data/common/genia-tagger-py
export PATH=$PATH:$PWD/scripts/data/common/geniass/
for d in "training" "testing";
do

    mkdir $intermediate_dir/${d}
    python $script_dir/gda2pubtator.py --input_folder $raw_dir/${d}_data/ \
                            --output_file $intermediate_dir/${d}.pubtator

    python $script_dir/process.py --input_file $intermediate_dir/${d}.pubtator \
                       --output_file $intermediate_dir/${d} \
                       --data GDA

    
done

mv $intermediate_dir/testing.data $intermediate_dir/test.data
mv $intermediate_dir/training.data $intermediate_dir/train+dev.data

python $script_dir/split_gda.py --input_file $intermediate_dir/train+dev.data \
                     --output_train $intermediate_dir/train.data \
                     --output_dev $intermediate_dir/dev.data \
                     --list $script_dir/train_gda_docs


python $script_dir/convert_gda2dygie.py --input_path $intermediate_dir/train.data --output_path $processed_dir/train.json --split train
python $script_dir/convert_gda2dygie.py --input_path $intermediate_dir/dev.data --output_path $processed_dir/dev.json --split dev
python $script_dir/convert_gda2dygie.py --input_path $intermediate_dir/test.data --output_path $processed_dir/test.json --split test