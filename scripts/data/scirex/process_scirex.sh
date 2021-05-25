raw_dir=data/scirex/raw-data
processed_dir=data/scirex/processed-data
script_dir=scripts/data/scirex

mkdir $processed_dir
python $script_dir/process_scirex.py --input_path $raw_dir/train.jsonl --output_path $processed_dir/train.json --split train
python $script_dir/process_scirex.py --input_path $raw_dir/dev.jsonl --output_path $processed_dir/dev.json --split dev
python $script_dir/process_scirex.py --input_path $raw_dir/test.jsonl --output_path $processed_dir/test.json --split test