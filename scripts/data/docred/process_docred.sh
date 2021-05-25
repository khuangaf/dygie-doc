out_dir=data/docred
log_dir=$out_dir/logs
raw_dir=$out_dir/raw-data
processed_dir=$out_dir/processed-data

devdev_list_path=$out_dir/dev-dev_dockey.list
devtest_list_path=$out_dir/dev-test_dockey.list
devdev_output_path=$processed_dir/devdev.json
devtest_output_path=$processed_dir/devtest.json

mkdir $out_dir
mkdir $raw_dir
mkdir $log_dir
mkdir $processed_dir

python scripts/data/docred/convert_docred2dygie.py --input_dir $raw_dir --output_dir $processed_dir

python scripts/data/docred/split_dev.py --dev_path $processed_dir/dev.json --devdev_list_path $devdev_list_path --devtest_list_path $devtest_list_path --devdev_output_path $devdev_output_path --devtest_output_path $devtest_output_path