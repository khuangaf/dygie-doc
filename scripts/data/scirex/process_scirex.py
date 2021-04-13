from typing import List, Dict, Any
import argparse
import json
from itertools import combinations

# copied over from https://github.com/allenai/SciREX/blob/master/scirex_utilities/entity_utils.py
available_entity_types_sciERC = ["Material", "Metric", "Task", "Generic", "OtherScientificTerm", "Method"]
map_available_entity_to_true = {"Material": "dataset", "Metric": "metric", "Task": "task", "Method": "model_name"}
map_true_entity_to_available = {v: k for k, v in map_available_entity_to_true.items()}
used_entities = list(map_available_entity_to_true.keys())


def scirex_span2dygie_span(span: List[int]):
    '''
    DyGIE has inclusive span
    '''
    return [span[0], span[1]-1]

def has_all_mentions(doc: Dict[str, Any], relation):
    '''
    Partially adapted from https://github.com/allenai/SciREX/blob/7daad660fe94f504433590b7a781cfabe1e179c6/scirex/evaluation_scripts/scirex_relation_evaluate.py
    Make sure each entity has at least one mention.
    '''
    has_mentions = all(len(doc["coref"][x[1]]) > 0 for x in relation)
    return has_mentions

def get_mentions(doc: Dict[str, Any], entity_name:str) -> List[List]:
    # dygie span boundary is inclusive
    return [scirex_span2dygie_span(span) for span in doc['coref'][entity_name]]

def generate_binary_relations(doc: Dict[str, Any]) -> List[List]:
    '''
    Break down 4-ary relations into binaries in order to train dygie.
    '''    
    binary_relations = []
    for types in combinations(used_entities, 2):
        relations = [tuple((t, x[t]) for t in types) for x in doc['n_ary_relations']]            

        # make sure each entity has at least one cluster and make (entity_1, entity_2, relation) unique
        relations = set([x for x in relations if has_all_mentions(doc, x)])
        
        for relation in relations:
            entity1, entity2 = relation
            entity1_type, entity1_name = entity1
            entity2_type, entity2_name = entity2
            
            #Each relation looks like this: (('Material', 'Quora_Question_Pairs'), ('Metric', 'Accuracy'))
            entity1_mentions = get_mentions(doc, entity1_name)
            entity2_mentions = get_mentions(doc, entity2_name)
            for entity1_mention in entity1_mentions:
                for entity2_mention in entity1_mentions:
                    binary_relations.append(entity1_mention + entity2_mention + ['has_relation'])
        
    return binary_relations

def process_ner(doc: Dict[str, Any]) -> List[List]:
    '''
    Put each NER annotation into sentences.
    '''
    def get_sentence_idx(sentence_boundaries: List[List], span: List[int]):
        for sentence_idx, sentence_boundary in enumerate(sentence_boundaries):
            if sentence_boundary[0] <= span[0] and sentence_boundary[1] >= span[1]:
                return sentence_idx
        print("Span crossing sentences")
        return None
        # raise ValueError((span, sentence_boundaries))

    sentence_boundaries = doc['sentences']
    res = [[] for _ in range(len(sentence_boundaries))]

    for ner in doc['ner']:
        span = ner[:2]
        entity_type = ner[2]
        sentence_idx = get_sentence_idx(sentence_boundaries, span)
        if sentence_idx is not None:
            res[sentence_idx].append(scirex_span2dygie_span(span) + [entity_type])

    return res
        
def process_sentences(doc: Dict[str, Any]) -> List[List]:
    res = []
    words = doc['words']
    for sentence_start, sentence_end in doc['sentences']:
        sentene_words = words[sentence_start:sentence_end]
        res.append(sentene_words)
    return res

def process_coref(doc: Dict[str, Any]) -> List[List]:
    '''
    Filter out cluster of size < 2.
    '''
    res = []
    for cluster in doc['coref'].values():
        if len(cluster) >= 2:
            res.append([scirex_span2dygie_span(span) for span in cluster])
    return res

def process_document(doc: Dict[str, Any]) -> Dict[str, Any]:

    document_relations = generate_binary_relations(doc)
    coref = process_coref(doc)
    ner = process_ner(doc)
    sentences = process_sentences(doc)
    return {
        'doc_key': doc['doc_id'],
        'dataset': 'scirex',
        'sentences': sentences,
        'ner': ner,
        'clusters': coref,
        'document_relations': document_relations,
        
    }

def process_file(input_path: str, output_path: str):

    with open(input_path, 'r') as f:
        input_data = [json.loads(l) for l in f.readlines()]
    
    processed_data = []
    for input_doc in input_data:
        processed_data.append(process_document(input_doc))
    
    with open(output_path, 'w') as f:
        for processed_doc in processed_data:
            f.write(json.dumps(processed_doc) + '\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    
    p.add_argument('--input_path', type=str) 
    p.add_argument('--output_path',type=str)
    args = p.parse_args()

    process_file(args.input_path, args.output_path)