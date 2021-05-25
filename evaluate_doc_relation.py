import argparse
from typing import Dict
from scipy.optimize import linear_sum_assignment
import json
import numpy as np
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("--gold-file")
parser.add_argument("--pred-file")


def phi_similarity(c1, c2):
    # entity level similarity function (phi_4 in CEAF paper)
    num_intersect = 0
    for m in c2:
        if m in c1:
            num_intersect += 1
    return 2 * num_intersect / (len(c1) + len(c2))

def ceaf(clusters, gold_clusters):
    '''
    Partially adapted from https://github.com/xinyadu/grit_doc_event_entity/blob/master/eval.py.
    '''
    # !!! need to comment the next line, the conll-2012 eval ignore singletons
    # clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi_similarity(gold_clusters[i], clusters[j])
    
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = sum(scores[row_ind, col_ind])
    assignments = zip(row_ind, col_ind)
    
    cluster_similarity = sum([scores[row, column] for row, column  in assignments])

    return cluster_similarity, zip(row_ind, col_ind) # need to re-zip again bc zip object disappear after iterating


def evaluate_relations(preds, golds):
    num_cluster_matched, num_cluster_predicted, num_cluster_gold = 0, 0, 0
    num_relation_matched, num_relation_predicted, num_relation_gold = 0, 0, 0

    for pred_doc, gold_doc in zip(preds, golds):
        pred_clusters = list(pred_doc['clusters'].values())
        gold_clusters = list(gold_doc['clusters'].values())
        pred_entity_names = list(pred_doc['clusters'].keys())
        gold_entity_names = list(gold_doc['clusters'].keys())

        cluster_similarity, assignments = ceaf(pred_clusters, gold_clusters)

        # cluster
        num_cluster_matched += cluster_similarity
        num_cluster_predicted += len(pred_clusters)
        num_cluster_gold += len(gold_clusters)
        
        # relation
        # TODO debug
        gold_entity_name2pred_entity_name = {gold_entity_names[gold_idx]: pred_entity_names[pred_idx] for gold_idx, pred_idx in assignments}
        
        # map from gold entity name to the aligned entity name
        mapped_gole_relations = [[gold_entity_name2pred_entity_name.get(rel[0]), gold_entity_name2pred_entity_name.get(rel[1]), rel[2]] for rel in gold_doc['relations']]
        
        for rel in mapped_gole_relations:
            if rel in pred_doc['relations']:
                num_relation_matched += 1                

        num_relation_predicted += len(pred_doc['relations'])
        num_relation_gold += len(gold_doc['relations'])

    cluster_p = num_cluster_matched / num_cluster_predicted if num_cluster_predicted != 0 else 0
    cluster_r = num_cluster_matched / num_cluster_gold if num_cluster_gold != 0 else 0 
    cluster_f1 = 2 * cluster_p * cluster_r / (cluster_p + cluster_r) if (cluster_p + cluster_r) != 0 else 0

    rel_p = num_relation_matched / num_relation_predicted if num_relation_predicted != 0 else 0
    rel_r = num_relation_matched / num_relation_gold if num_relation_gold != 0 else 0
    rel_f1 = 2 * rel_p * rel_r / (rel_p + rel_r) if (rel_p + rel_r) != 0 else 0
    return {
        'cluster_p': cluster_p,
        'cluster_r': cluster_r,
        'cluster_f1': cluster_f1,
        'relation_p': rel_p,
        'relation_r': rel_r,
        'relation_f1': rel_f1,
    }



def main(args):
    def load_jsonl(path):
        with open(path, 'r') as f:
            res = [json.loads(l) for l in f.readlines()]
        return res
    preds = load_jsonl(args.pred_file)
    golds = load_jsonl(args.gold_file)

    print(evaluate_relations(golds, preds))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)