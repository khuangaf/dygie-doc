echo "Processing LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2"
python scripts/data/ere/preprocess_ere.py -i data/ere/raw-data/LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2/data -o data/ere/processed-data -d normal
echo "Processing LDC2015E68_DEFT_Rich_ERE_English_Training_Annotation_R2_V2"
python scripts/data/ere/preprocess_ere.py -i data/ere/raw-data/LDC2015E68_DEFT_Rich_ERE_English_Training_Annotation_R2_V2/data -o data/ere/processed-data -d r2v2
echo "Processing LDC2015E78_DEFT_Rich_ERE_Chinese_and_English_Parallel_Annotation_V2"
python scripts/data/ere/preprocess_ere.py -i data/ere/raw-data/LDC2015E78_DEFT_Rich_ERE_Chinese_and_English_Parallel_Annotation_V2/data -o data/ere/processed-data -d parallel

python scripts/data/ere/split_ere.py -i data/ere/processed-data/normal.json data/ere/processed-data/r2v2.json data/ere/processed-data/parallel.json -o data/ere/processed-data/ -s data/ere/splits/