out_dir=data/cdr
raw_dir=$out_dir/raw-data

mkdir $out_dir
mkdir $raw_dir

wget https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip -P $raw_dir
unzip $raw_dir/CDR_Data.zip -d $raw_dir
rm $raw_dir/CDR_Data.zip 
