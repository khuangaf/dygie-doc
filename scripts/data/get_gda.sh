out_dir=data/gda
raw_dir=$out_dir/raw-data

mkdir $out_dir

wget https://bitbucket.org/alexwuhkucs/gda-extraction/get/fd4a7409365e.zip -P $out_dir
unzip $out_dir/fd4a7409365e.zip -d $out_dir
mv $out_dir/alexwuhkucs-gda-extraction-fd4a7409365e $raw_dir
rm $out_dir/fd4a7409365e.zip 
