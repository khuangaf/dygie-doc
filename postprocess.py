import argparse
import json
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import operator


parser = argparse.ArgumentParser()
parser.add_argument("--is-gold", action="store_true", help="Whether the input file is gold standard." )
parser.add_argument("--input-path", help="Path to the input file to be converted to entity-level representation.")
parser.add_argument("--output-path", help="Path to store the data in entity-level representation.")



def convert(doc, is_gold):
    if is_gold:
        prefix = ''
    else:
        prefix = 'predicted_'
    
    doc_tokens = [tok for sent in doc['sentences'] for tok in sent]
    cluster_dict = {}
    span2cluster : Dict[Tuple[int], str]= {} 
    cluster_relation_count: Dict[Tuple(str), Dict[str, int]] = defaultdict(Counter)
    entity_name_dict = {}
    cluster_in_relation = set()
    # gather cluster
    for cluster in doc[f'{prefix}clusters']:
        longest_mention_span = sorted(cluster, key=lambda x:x[1]-x[0], reverse=True)[0]
        # entity_key = ' '.join(doc_tokens[longest_mention_span[0]:longest_mention_span[1]+1])
        entity_key = str(len(cluster_dict))
        entity_name_dict[entity_key] = ' '.join(doc_tokens[longest_mention_span[0]:longest_mention_span[1]+1])
        
        is_merged_entity = False
        # merge mention with other entity if mention string is the same. do this for prediction only
        if not is_gold and len(cluster)==1:
            span = cluster[0]
            mention_string = ' '.join(doc_tokens[span[0]:span[1]+1])
            for other_span, other_cluster_id in span2cluster.items():
                other_mention_string = ' '.join(doc_tokens[other_span[0]:other_span[1]+1])
                if mention_string.lower() == other_mention_string.lower():
                    entity_key = other_cluster_id
                    is_merged_entity = True
                    span2cluster[tuple(span)] = entity_key
                    cluster_dict[entity_key].append(span)        
                    break
        # only add treat this entity to span2cluster and cluster_dict if it is not merged.
        if not is_merged_entity:
            for span in cluster:
                span2cluster[tuple(span)] = entity_key
            cluster_dict[entity_key] = cluster

                
    # convert relation to entity level
    entity_level_relations = []
    for relation in doc[f'{prefix}document_relations']:
        idx1 = str(relation[0])
        idx2 = str(relation[1])
        relation_type = relation[2]
        
        if idx1 in cluster_dict and idx2 in cluster_dict:
            entity_level_relations
            entity_level_relations.append([idx1, idx2, relation_type])

    
        
    


    return {
        'doc_key': doc['doc_key'],
        'clusters': cluster_dict,
        'relations': entity_level_relations,
        'entity_names': entity_name_dict
    }

def main(args):
    def load_jsonl(path):
        with open(path, 'r') as f:
            res = [json.loads(l) for l in f.readlines()]
        return res
    mention_level_data = load_jsonl(args.input_path)

    entity_level_data = [convert(doc, args.is_gold) for doc in mention_level_data]

    with open(args.output_path, 'w') as f:
        for doc in entity_level_data:
            f.write(json.dumps(doc)+'\n')
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)    