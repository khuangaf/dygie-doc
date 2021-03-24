import json
import os
import argparse

def split_data(input_files, output_dir, split_path):
    """Splits the input file into train/dev/test sets.

    Args:
        input_file (str): path to the input file.
        output_dir (str): path to the output directory.
        split_path (str): path to the split directory that contains three files,
            train.doc.txt, dev.doc.txt, and test.doc.txt . Each line in these
            files is a document ID.
    """
    print('Splitting the dataset into train/dev/test sets')
    train_docs, dev_docs, test_docs = set(), set(), set()
    # Load doc ids
    with open(os.path.join(split_path, 'train.doc.txt')) as r:
        train_docs.update(r.read().strip('\n').split('\n'))
    with open(os.path.join(split_path, 'dev.doc.txt')) as r:
        dev_docs.update(r.read().strip('\n').split('\n'))
    with open(os.path.join(split_path, 'test.doc.txt')) as r:
        test_docs.update(r.read().strip('\n').split('\n'))
    
    all_lines = []
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as r:
            all_lines += r.readlines()
    # Split the dataset
    with open(os.path.join(output_dir, 'train.json'), 'w') as w_train, \
        open(os.path.join(output_dir, 'dev.json'), 'w') as w_dev, \
        open(os.path.join(output_dir, 'test.json'), 'w') as w_test:
        for line in all_lines:
            inst = json.loads(line)
            doc_id = inst['doc_key']
            if doc_id in train_docs:
                w_train.write(line)
            elif doc_id in dev_docs:
                w_dev.write(line)
            elif doc_id in test_docs:
                w_test.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input_files',
                        required=True,
                        nargs='+',
                        help='Path to the input json files')

    parser.add_argument('-o',
                        '--output',
                        required=True,
                        help='Path to the output folder')                        
    parser.add_argument('-s',
                        '--split',
                        required=True,
                        help='Path to the split folder')
    
    args = parser.parse_args()

    split_data(args.input_files, args.output, args.split)