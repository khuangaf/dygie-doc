out_dir=data/docred
log_dir=$out_dir/logs
raw_dir=$out_dir/raw-data
processed_dir=$out_dir/processed-data

mkdir $out_dir
mkdir $raw_dir
mkdir $log_dir
mkdir $processed_dir

# specify split and corresponding gdrive file id
declare -A split2FILEID
split2FILEID[train]=1NN33RzyETbanw4Dg2sRrhckhWpzuBQS9
split2FILEID[dev]=1fDmfUUo5G7gfaoqWWvK81u08m71TK2g7
split2FILEID[test]=1lAVDcD94Sigx7gR3jTfStI66o86cflum

# download data from gdrive
for split in "${!split2FILEID[@]}"
do
    FILE=$raw_dir/$split.json
    FILEID=${split2FILEID[$split]}
    # Adapted partly from Wasi's code. (UCLA PhD)
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${FILEID}" > ./tmp
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${FILEID}" -o ${FILE}
    rm ./cookie
    rm ./tmp
done

python scripts/data/docred/convert_docred2dygie.py --input_dir $raw_dir --output_dir $processed_dir