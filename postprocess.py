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
    # gather cluster
    for cluster in doc[f'{prefix}clusters']:
        longest_mention_span = sorted(cluster, key=lambda x:x[1]-x[0], reverse=True)[0]
        # entity_key = ' '.join(doc_tokens[longest_mention_span[0]:longest_mention_span[1]+1])
        entity_key = str(len(cluster_dict))
        entity_name_dict[entity_key] = ' '.join(doc_tokens[longest_mention_span[0]:longest_mention_span[1]+1])
        cluster_dict[entity_key] = cluster
        for span in cluster:
            span2cluster[tuple(span)] = entity_key
    
    # gather mention that doesn't have any coref mention
    for entities in doc[f'{prefix}ner']:
        for mention in entities:
            span = mention[:2]
            # entity_key = ' '.join(doc_tokens[span[0]:span[1]+1])
            entity_key = str(len(cluster_dict))
            entity_name_dict[entity_key] = ' '.join(doc_tokens[span[0]:span[1]+1])
            if tuple(span) not in span2cluster:
    
                # If such entity string already exist, do not replace.
                # TODO: we might come up a better way to handle this.
                if entity_key not in cluster_dict:
                    cluster_dict[entity_key] = [span]
                    span2cluster[tuple(span)] = entity_key
    
    # convert relation to entity level
    for relation in doc[f'{prefix}document_relations']:
        span1 = tuple(relation[:2])
        span2 = tuple(relation[2:4])
        relation_type = relation[4]
        if span1 in span2cluster and span2 in span2cluster:
            entity1 = span2cluster[span1]
            entity2 = span2cluster[span2]
            cluster_relation_count[(entity1, entity2)][relation_type] += 1

    entity_level_relations = []
    for (entity1, entity2), relation_count in cluster_relation_count.items():
        # get argmax
        best_relation_type = max(relation_count.items(), key=operator.itemgetter(1))[0]
        entity_level_relations.append([entity1, entity2, best_relation_type])

    # TODO: maybe should consider not to include entities that does not involve in any relation.
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