out_dir=data/scirex
raw_dir=$out_dir/raw-data
processed_dir=$out_dir/processed-data

mkdir $out_dir
mkdir $raw_dir
mkdir $log_dir
mkdir $processed_dir

wget https://github.com/allenai/SciREX/raw/master/scirex_dataset/release_data.tar.gz -P $raw_dir
tar xf $raw_dir/release_data.tar.gz -C $raw_dir
mv $raw_dir/release_data/* $raw_dir


rm -rf $raw_dir/release_data/
rm $raw_dir/release_data.tar.gz
