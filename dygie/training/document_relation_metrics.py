from overrides import overrides
from typing import Dict, Any
# from dygie.data.dataset_readers.document import Document
from dygie.training.relation_metrics import RelationMetrics

from dygie.training.f1 import compute_f1


class DocumentRelationMetrics(RelationMetrics):
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold spans.
    """
    def __init__(self):
        self.reset()

    # TODO(dwadden) This requires decoding because the dataset reader gets rid of gold spans wider
    # than the span width. So, I can't just compare the tensor of gold labels to the tensor of
    # predicted labels.
    @overrides
    def __call__(self, predicted_relations: Dict[Any, Any], document):
        
        gold_relations = document.document_relation_dict
        self._total_gold += len(gold_relations)
        self._total_predicted += len(predicted_relations)
        
        for (span_1, span_2), label in predicted_relations.items():
            ix = (span_1, span_2)
            if ix in gold_relations and gold_relations[ix] == label:
                self._total_matched += 1

    
