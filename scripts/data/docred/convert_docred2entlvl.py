import pandas as pd
import json
import numpy as np
import argparse
import os

def compute_sentence_starts(sentences):
    return np.cumsum([0] + [len(sent) for sent in sentences]).tolist()

def convert_doc_relations(labels, vertex_sets, sentences):
    '''
    Each element vertex_sets represent an entity. We add relation labels for each pair of mention.
    Output: A list of relations that correspond to an entire document.
    '''
    doc_relations = []

    
    for label in labels:
        '''
        A label looks like this {'r': 'P159', 'h': 0, 't': 2, 'evidence': [0]}
        '''
        relation_label = label['r']
        head_entity_idx = str(label['h'])
        tail_entity_idx = str(label['t'])
        
        doc_relations.append([head_entity_idx, tail_entity_idx, relation_label])
    return doc_relations

def convert_ner(vertex_sets, sentences):

    # create a nested list with the same size as the length of sentences 
    ner = [[] for _ in range(len(sentences))]

    # we need this to compute token offsets w.r.t. document
    sentence_starts = compute_sentence_starts(sentences)

    # each vertex_sets is an entity
    for entity in vertex_sets:
        for mention in entity:
            sent_id = mention['sent_id']
            ner_label = mention['type']
            mention_start = mention['pos'][0] + sentence_starts[sent_id]
            mention_end = mention['pos'][1] + sentence_starts[sent_id] - 1 

            # append the mention to corresponding sentence
            ner[sent_id].append([mention_start, mention_end, ner_label])

    return ner


def convert_coreference(vertex_sets, sentences):
    
    clusters = {}

    # we need this to compute token offsets w.r.t. document
    sentence_starts = compute_sentence_starts(sentences)

    # each vertex_sets is an entity
    for ent_id, entity in enumerate(vertex_sets):
        ent_id = str(ent_id)
        cluster = []
        for mention in entity:
            sent_id = mention['sent_id']
            
            mention_start = mention['pos'][0] + sentence_starts[sent_id]
            mention_end = mention['pos'][1] + sentence_starts[sent_id] - 1 

            cluster.append([mention_start, mention_end])
        
        # a cluster must have 2 or more mentions
        # if len(cluster) >= 2:
        clusters[ent_id] = cluster
    return clusters

def write_to_jsonl(process_documents, output_path):
    with open(output_path, 'w') as f:
        for process_document in process_documents:
            f.write(json.dumps(process_document) + '\n')


def process_document(input_document, split, idx):
    '''
    Process a document in docred format. split and idx are only for constructing doc_key
    
    '''
    res = {
        'doc_key': f'{split}-{idx}',
        'sentences': input_document['sents'],
    }
    # labels can be empty, use .get to work around
    res['relations'] = convert_doc_relations(input_document.get('labels',[]), input_document['vertexSet'], input_document['sents'])
    
    res['clusters'] = convert_coreference(input_document['vertexSet'], input_document['sents'])

    return res

def process_splits(input_dir, output_dir):
    
    for split in ['train','dev']:

        input_path = os.path.join(input_dir, f'{split}.json')

        processed_documents = []
        with open(input_path, 'r') as f:
            input_documents = json.load(f) 

            
        for idx, input_document in enumerate(input_documents):
            processed = process_document(input_document, split, idx)
            processed_documents.append(processed)
        
        output_path = os.path.join(output_dir, f'{split}.entlvl.json')
        write_to_jsonl(processed_documents, output_path) 



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    
    p.add_argument('--input_dir', type=str, help="input patters that will be passed to glob to fetch file names.") 
    p.add_argument('--output_dir',type=str, help="path to store the output json file.")
    args = p.parse_args()

    process_splits(args.input_dir, args.output_dir)