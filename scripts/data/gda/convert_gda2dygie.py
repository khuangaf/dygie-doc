import argparse
from recordtype import recordtype
import json

EntityInfo = recordtype('EntityInfo', 'type mstart mend sentNo')
PairInfo = recordtype('PairInfo', 'type direction cross closeA closeB')

def chunks(l, n):
    """
    Copied from statistics.py
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        yield l[i:i + n]

def convert_document(line, split):
    res = {
        'ner':[],
        'document_relations':[],
        'clusters':[],
        '_split':split
    }
    line = line.rstrip().split('\t')
    
    
    pairs = chunks(line[2:], 17)
    res['doc_key'] = line[0]
    res['sentences'] = [sentence.split(' ') for sentence in line[1].split('|') ]
    num_sentences = len(line[1].split('|'))
    
    res['ner'] = [[] for _ in range(num_sentences)]
    entities, relations = {}, {}
    entity_id2idx = {} # map from original entity_id to the index of the entity for document_relations
    for p in pairs:
        # pairs
        if (p[5], p[11]) not in relations:
            relations[(p[5], p[11])] = PairInfo(p[0], p[1], p[2], p[3], p[4])
        else:
            print('duplicates!')

        
        if p[5] not in entities:
            entities[p[5]] = EntityInfo(p[7], p[8], p[9], p[10])

        if p[11] not in entities:
            entities[p[11]] = EntityInfo(p[13], p[14], p[15], p[16])

    # gather ner, relation, and coreference information
    for entity_id, entity in entities.items():
        entity_type = entity.type
        cluster = []
        for mention_start, mention_end, sentence_num in zip(entity.mstart.split(':'), entity.mend.split(':'), entity.sentNo.split(':')):
            
            res['ner'][int(sentence_num)].append([int(mention_start), int(mention_end)-1, entity_type])
            cluster.append([int(mention_start), int(mention_end)-1])
        # an entity cluster must have >= 2 mentions
        # if len(cluster) >= 2:
        
        entity_id2idx[entity_id] = len(res['clusters'])
        res['clusters'].append(cluster)

    for (entity1_id, entity2_id), relation in relations.items():
        relation_type = relation.type
        if relation_type in ['1:NR:2','not_include']: continue # skip negative relation
        if relation.direction == 'L2R':
            entity1 = entities[entity1_id]
            entity2 = entities[entity2_id]
            entity1_idx = entity_id2idx[entity1_id]
            entity2_idx = entity_id2idx[entity2_id]
        elif relation.direction == 'R2L':
            entity1 = entities[entity2_id]
            entity2 = entities[entity1_id]
            entity1_idx = entity_id2idx[entity2_id]
            entity2_idx = entity_id2idx[entity1_id]
        else:
            raise ValueError(f"Unexpected relation direction {relation.direction}")
        # take all the combination of all mentions 
        for mention1_start, mention1_end in zip(entity1.mstart.split(':'), entity1.mend.split(':')):
            for mention2_start, mention2_end in zip(entity2.mstart.split(':'), entity2.mend.split(':')):
                res['document_relations'].append([int(mention1_start), int(mention1_end)-1, int(mention2_start), int(mention2_end)-1, relation_type])
        # res['document_relations'].append([entity1_idx, entity2_idx, relation_type])

    return res

def convert(input_path, output_path, split):
    res = []
    
    with open(input_path, 'r') as infile:
        for line in infile:
            res.append(convert_document(line, split))
    
    with open(output_path, 'w') as f:
        for doc in res:
            f.write(json.dumps(doc)+'\n')
    





if __name__ == '__main__':
    p = argparse.ArgumentParser()
    
    p.add_argument('--input_path', type=str, help="") 
    p.add_argument('--output_path',type=str, help="path to store the output json file.")
    p.add_argument('--split',type=str, help="train/dev/test.")
    args = p.parse_args()

    convert(args.input_path, args.output_path, args.split)