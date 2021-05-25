import pandas as pd
import json
import numpy as np
import argparse
import os

def split(dev_path, devdev_list_path, devtest_list_path, devdev_output_path, devtest_output_path):

    with open(dev_path, 'r') as f:
        dev = [json.loads(l) for l in f.readlines()]

    with open(devdev_list_path, 'r') as f:
        devdev_dockey = [l.strip() for l in f.readlines()]

    with open(devtest_list_path, 'r') as f:
        devtest_dockey = [l.strip() for l in f.readlines()]

    devdev = [doc for doc in dev if doc['doc_key'] in devdev_dockey]
    devtest = [doc for doc in dev if doc['doc_key'] in devtest_dockey]

    assert len(devdev)  == len(devdev) == 500

    with open(devdev_output_path, 'w') as f:
        for doc in devdev:
            f.write(json.dumps(doc)+'\n')

    with open(devtest_output_path, 'w') as f:
        for doc in devtest:
            f.write(json.dumps(doc)+'\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    
    p.add_argument('--dev_path', type=str, help="the path to the path file") 
    p.add_argument('--devdev_list_path', type=str, help="the path to the path file") 
    p.add_argument('--devtest_list_path', type=str, help="the path to the path file")
    p.add_argument('--devdev_output_path', type=str, help="the path to the path file")  
    p.add_argument('--devtest_output_path', type=str, help="the path to the path file")  
    args = p.parse_args()
    split(args.dev_path, args.devdev_list_path, args.devtest_list_path, args.devdev_output_path, args.devtest_output_path)

    