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
    
    # we need this to compute token offsets w.r.t. document
    sentence_starts = compute_sentence_starts(sentences)
    
    for label in labels:
        '''
        A label looks like this {'r': 'P159', 'h': 0, 't': 2, 'evidence': [0]}
        '''
        relation_label = label['r']
        head_entity_idx = label['h']
        tail_entity_idx = label['t']
        for head_mention in vertex_sets[head_entity_idx]:
            for tail_mention in vertex_sets[tail_entity_idx]:
                '''
                A mention looks like this {'pos': [0, 4], 'type': 'ORG', 'sent_id': 0, 'name': 'Zest Airways, Inc.'}
                '''
                
                
                span_1_sent_offset = sentence_starts[head_mention['sent_id']]
                span_2_sent_offset = sentence_starts[tail_mention['sent_id']]

                span_1_start = head_mention['pos'][0] + span_1_sent_offset
                span_1_end = head_mention['pos'][1] + span_1_sent_offset -1 
                
                span_2_start = tail_mention['pos'][0] + span_2_sent_offset
                span_2_end = tail_mention['pos'][1] + span_2_sent_offset -1                 
                #[start_tok_1, end_tok_1, start_tok_2, end_tok_2, label]
                doc_relations.append([span_1_start, span_1_end, span_2_start, span_2_end, relation_label])
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
    
    clusters = []

    # we need this to compute token offsets w.r.t. document
    sentence_starts = compute_sentence_starts(sentences)

    # each vertex_sets is an entity
    for entity in vertex_sets:
        cluster = []
        for mention in entity:
            sent_id = mention['sent_id']
            
            mention_start = mention['pos'][0] + sentence_starts[sent_id]
            mention_end = mention['pos'][1] + sentence_starts[sent_id] - 1 

            cluster.append([mention_start, mention_end])
        
        # a cluster must have 2 or more mentions
        if len(cluster) >= 2:
            clusters.append(cluster)
    return clusters

def write_to_jsonl(process_documents, output_path):
    with open(output_path, 'w') as f:
        for process_document in process_documents:
            f.write(json.dumps(process_document) + '\n')


def process_document(input_document, split, idx):
    '''

    '''
    res = {
        'doc_key': f'{split}-{idx}',
        'sentences': input_document['sents']
    }
    # labels can be empty, use .get to work around
    res['doc_relations'] = convert_doc_relations(input_document.get('labels',[]), input_document['vertexSet'], input_document['sents'])
    res['ner'] = convert_ner(input_document['vertexSet'], input_document['sents'])
    res['clusters'] = convert_coreference(input_document['vertexSet'], input_document['sents'])

    return res

def process_splits(input_dir, output_dir):
    
    for split in ['train','dev','test']:

        input_path = os.path.join(input_dir, f'{split}.json')

        processed_documents = []
        with open(input_path, 'r') as f:
            input_documents = json.load(f) 

            
        for idx, input_document in enumerate(input_documents):
            processed = process_document(input_document, split, idx)
            processed_documents.append(processed)
        
        output_path = os.path.join(output_dir, f'{split}.json')
        write_to_jsonl(processed_documents, output_path)


        

        
    





if __name__ == '__main__':
    p = argparse.ArgumentParser()
    
    p.add_argument('--input_dir', type=str, help="input patters that will be passed to glob to fetch file names.") 
    p.add_argument('--output_dir',type=str, help="path to store the output json file.")
    args = p.parse_args()

    process_splits(args.input_dir, args.output_dir)